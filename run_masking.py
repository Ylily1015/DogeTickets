# -*- coding: utf-8 -*-

import os
import re
import time
import math
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import AdamW, get_scheduler

from tqdm.auto import tqdm

from data import get_reader_class, get_pipeline_class, Dataset, DistributedDataset
from metrics import get_metric_fn
from models import get_model_class
from utils import set_seed, add_kwargs_to_config, keep_recent_ckpt, Logger, AverageMeter
from mask_utils import *


logger = Logger()


def gather(tensor, num_examples):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    output = concat[:num_examples] # Truncate dummy elements added by DistributedSampler.
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a classification task.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",    
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The task to train on, for indexing data reader.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Where to load a glue dataset.",
    )
    parser.add_argument(
        "--train_data_domains",
        nargs='+', 
        type=str,
        default=None,
        help="train data domains",
    )
    parser.add_argument(
        "--test_data_domains",
        nargs='+', 
        type=str,
        default=None,
        help="test data domains"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--init_model_path", 
        type=str, 
        default="", 
        help="Where to store the init model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training loader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument("--lam", type=float, default=0, help="Lambda for avg/var trade-off")
    parser.add_argument("--seed", type=int, default=776, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 or not.")
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU or not.")
    parser.add_argument("--do_mask", action="store_true", help ="Do mask or not.")
    parser.add_argument("--do_mask_with_domain", action="store_true", help ="Do mask with domain information or not.")
    parser.add_argument('--normalize_by_layer', action='store_true')
    parser.add_argument('--percent_seq', default=[50], nargs="*", type=float, help="Mask percent list")
    parser.add_argument('--min_attentionum_heads', default=1, type=int, help="Minmum number of attetion heads.")
    parser.add_argument('--mask_reverse', action="store_true", help="Mask reverse or not.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.do_mask:
        args.output_dir = os.path.join(args.output_dir, args.model_type, args.task_name, "mask")
    else:
        args.output_dir = os.path.join(args.output_dir, args.model_type, args.task_name, f"mask-domain-{args.lam}")
     
    os.makedirs(args.output_dir, exist_ok=True)
    args.data_dir = os.path.join(args.data_dir, args.task_name)
            
    is_dist = (args.local_rank != -1)
    is_main = (args.local_rank == -1 or args.local_rank == 0)
    is_fp16 = is_dist and args.use_fp16
    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    if is_dist:
        # Initialize DDP.
        dist.init_process_group(backend='nccl')
        # Pin GPU to be used to process local rank (one GPU per process).
        torch.cuda.set_device(args.local_rank)

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.add_stream_handler()
    logger.add_file_handler(args.output_dir)
    if is_main:
        logger.set_verbosity_info() 
    else:
        logger.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load metric functin and data reader.
    metric_fn = get_metric_fn(args.task_name)
    data_reader = get_reader_class(args.task_name)(args.data_dir)
    label_map, reverse_label_map, num_labels = data_reader.get_label_map()
    domain_map = data_reader.get_domain_map()

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    if args.do_mask:
        logger.info("***** Computing the head and ffn mask *****")

        model_path = args.model_name_or_path

        # Load pretrained tokenizer with necessary resizing.
        tokenizer = tokenizer_class.from_pretrained(model_path)
        
        # Data pipeline.
        data_pipeline = pipeline_class(tokenizer, label_map, domain_map, args.max_length)
        
        config = config_class.from_pretrained(model_path)
        model = model_class.from_pretrained(
            model_path,
            config=config,
        )
        model = model.to(device)

        train_examples = data_reader.get_train_examples(domains=args.train_data_domains)
        train_examples = data_pipeline.build(train_examples)
        train_dataset = Dataset(train_examples, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_pipeline.collate)
        
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers

        # Calculate importance scores for each task at each layer
        # head_importance: {layer_idx:[score_list]}
        head_importance, ffn_importance = calculate_importance(model, train_loader, num_heads=num_heads, num_layers=num_layers, normalize_by_layer=args.normalize_by_layer)
        save_importance(args.output_dir, head_importance, ffn_importance)
        logger.info(f"  Head importance scores :")
        for layer in range(num_layers):
            layer_scores = head_importance[layer].cpu().data
            
            logger.info("\t".join(f"{x:.5f}" for x in layer_scores))

        logger.info(f"  FFN importance scores :")
        logger.info("\t".join(f"{x:.5f}" for x in ffn_importance))

        # Generate head mask json file.
        # Determine the sequence of heads to mask.
        num_head_seq = get_head_sequence(args.percent_seq, args.min_attentionum_heads, num_heads=num_heads, num_layers=num_layers)
        heads_to_mask = {}
        total_masked = 0
        for step, num_to_mask in enumerate(num_head_seq):
            # head_to_mask: {layer_idx:set(head_idx)}
            heads_to_mask = what_to_mask(head_importance, num_to_mask, args.min_attentionum_heads, heads_to_mask, num_heads=num_heads, num_layers=num_layers, mask_reverse=args.mask_reverse)
            total_masked += num_to_mask

            logger.info(f"  Number of heads to be masked: {total_masked}")
            logger.info(f"  {heads_to_mask}")

            if not args.mask_reverse:
                head_mask_dir = os.path.join(args.output_dir, "mask_{}_heads".format(total_masked))
            else:
                head_mask_dir = os.path.join(args.output_dir, "reverse_mask_{}_heads".format(total_masked))

            os.makedirs(head_mask_dir, exist_ok=True) 
            head_mask_file = os.path.join(head_mask_dir, "heads_to_mask.json")   
            dump_head_mask_file(head_mask_file, heads_to_mask)

        # Generate ffn mask json file.
        ffns_to_mask = torch.argsort(torch.abs(ffn_importance)).cpu().tolist()
        for layer_idx in range(num_layers):
            logger.info(f"  Number of ffn to be masked: {layer_idx+1}")
            if not args.mask_reverse:
                logger.info(f"  {ffns_to_mask[:layer_idx+1]}")
                ffn_mask_dir = os.path.join(args.output_dir, "mask_{}_ffns".format(layer_idx+1))
            else:
                logger.info(f"  {ffns_to_mask[-(layer_idx+1):]}")
                ffn_mask_dir = os.path.join(args.output_dir, "reverse_mask_{}_ffns".format(layer_idx+1))

            os.makedirs(ffn_mask_dir, exist_ok=True)
            ffn_mask_file = os.path.join(ffn_mask_dir, "ffns_to_mask.txt")
            if not args.mask_reverse:
                ffn_mask = ffns_to_mask[:layer_idx+1]
            else:
                ffn_mask = ffns_to_mask[-(layer_idx+1):]
            dump_ffn_mask_file(ffn_mask_file, ffn_mask)
            
    if args.do_mask_with_domain:
        logger.info("***** Computing the head and ffn mask *****")

        try:
            model_path = best_dev_path
        except:
            model_path = args.model_name_or_path

        # Load pretrained tokenizer with necessary resizing.
        tokenizer = tokenizer_class.from_pretrained(model_path)
        
        # Data pipeline.
        data_pipeline = pipeline_class(tokenizer, label_map, domain_map, args.max_length)
        
        config = config_class.from_pretrained(model_path)
        model = model_class.from_pretrained(
            model_path,
            config=config,
        )
        model = model.to(device)

        train_examples = {}
        train_dataset = {}
        train_loader = {}
        for domain in args.train_data_domains:
            train_examples[domain] = data_reader.get_train_examples(domains=[domain])
            train_examples[domain] = data_pipeline.build(train_examples[domain])
            train_dataset[domain] = Dataset(train_examples[domain], shuffle=False)
            train_loader[domain] = DataLoader(train_dataset[domain], batch_size=args.per_device_train_batch_size, collate_fn=data_pipeline.collate)
        
        num_heads = config.num_attention_heads
        num_layers = config.num_hidden_layers

        # Calculate importance scores for each task at each layer.
        # head_importance: {layer_idx:[score_list]}
        head_domain_importance, ffn_domain_importance = calculate_importance(model, train_loader, num_heads=num_heads, num_layers=num_layers, normalize_by_layer=args.normalize_by_layer, domains=args.train_data_domains)
        for domain in args.train_data_domains:
            logger.info(f"{domain} Head importance scores :")
            for layer in range(num_layers):
                layer_scores = head_domain_importance[domain][layer].cpu().data
                logger.info("\t".join(f"{x:.5f}" for x in layer_scores))

            logger.info(f"{domain}  FFN importance scores :")
            logger.info("\t".join(f"{x:.5f}" for x in ffn_domain_importance[domain]))
        save_importance(args.output_dir, head_domain_importance, ffn_domain_importance, domains=args.train_data_domains)

        head_importance, ffn_importance = average_domain_importance(args.train_data_domains, head_domain_importance, ffn_domain_importance, args.lam, device, num_heads, num_layers)
        save_importance(args.output_dir, head_importance, ffn_importance)

        # Generate head mask json file.
        # Determine the sequence of heads to mask.
        num_head_seq = get_head_sequence(args.percent_seq, args.min_attentionum_heads, num_heads=num_heads, num_layers=num_layers)
        heads_to_mask = {}
        total_masked = 0
        for step, num_to_mask in enumerate(num_head_seq):
            # head_to_mask: {layer_idx:set(head_idx)}
            heads_to_mask = what_to_mask(head_importance, num_to_mask, args.min_attentionum_heads, heads_to_mask, num_heads=num_heads, num_layers=num_layers, mask_reverse=args.mask_reverse)
            total_masked += num_to_mask

            logger.info(f"  Number of heads to be masked: {total_masked}")
            logger.info(f"  {heads_to_mask}")

            if not args.mask_reverse:
                head_mask_dir = os.path.join(args.output_dir, "mask_{}_heads".format(total_masked))
            else:
                head_mask_dir = os.path.join(args.output_dir, "reverse_mask_{}_heads".format(total_masked))

            os.makedirs(head_mask_dir, exist_ok=True) 
            head_mask_file = os.path.join(head_mask_dir, "heads_to_mask.json")   
            dump_head_mask_file(head_mask_file, heads_to_mask)
        

        # Generate ffn mask json file.
        ffns_to_mask = torch.argsort(torch.abs(ffn_importance)).cpu().tolist()
        for layer_idx in range(num_layers):
            logger.info(f"  Number of ffn to be masked: {layer_idx+1}")
            if not args.mask_reverse:
                logger.info(f"  {ffns_to_mask[:layer_idx+1]}")
                ffn_mask_dir = os.path.join(args.output_dir, "mask_{}_ffns".format(layer_idx+1))
            else:
                logger.info(f"  {ffns_to_mask[-(layer_idx+1):]}")
                ffn_mask_dir = os.path.join(args.output_dir, "reverse_mask_{}_ffns".format(layer_idx+1))

            os.makedirs(ffn_mask_dir, exist_ok=True)
            ffn_mask_file = os.path.join(ffn_mask_dir, "ffns_to_mask.txt")
            if not args.mask_reverse:
                ffn_mask = ffns_to_mask[:layer_idx+1]
            else:
                ffn_mask = ffns_to_mask[-(layer_idx+1):]
            dump_ffn_mask_file(ffn_mask_file, ffn_mask)
    

if __name__ == "__main__":
    """
    1. Single-Node multi-process distributed training

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
                arguments of your training script)

    2. Multi-Node multi-process distributed training: (e.g. two nodes)


    Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)

    Node 2:

    ::

        >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
                --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
                --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
                and all other arguments of your training script)
    """
    main()