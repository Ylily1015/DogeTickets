# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np

import torch
import pdb

from utils import Logger

logger = Logger()

def average_domain_importance(domains, head_importance, ffn_importance, lam, device, num_heads=0, num_layers=0,):
    domain_num = len(domains)
    
    avg_head_ipt = torch.zeros((num_layers, num_heads), device=device)
    var_head_ipt = torch.zeros((num_layers, num_heads), device=device)
    avg_ffn_ipt = torch.zeros(num_layers, device=device)
    var_ffn_ipt = torch.zeros(num_layers, device=device)

    for domain in domains:
        avg_head_ipt += head_importance[domain] / domain_num
        avg_ffn_ipt += ffn_importance[domain] / domain_num
    
    for domain in domains:
        var_head_ipt += torch.pow((head_importance[domain]-avg_head_ipt),2) / domain_num
        var_ffn_ipt += torch.pow((ffn_importance[domain]-avg_ffn_ipt),2) / domain_num
    
    final_head_ipt = avg_head_ipt - lam * var_head_ipt
    final_ffn_ipt = avg_ffn_ipt - lam * var_ffn_ipt
    
    return final_head_ipt, final_ffn_ipt

def calculate_importance(model, dataloaders, num_heads=0, num_layers=0, normalize_by_layer=True, domains=None):
    if domains is not None:
        head_importance = {}
        ffn_importance = {}
        for domain in domains:
            head_importance[domain] = torch.zeros(num_layers, num_heads).to(model.device)
            ffn_importance[domain] = torch.zeros(num_layers).to(model.device)

        model.eval()
        for domain in domains:
            total_len = 0
            for batch in dataloaders[domain]:
                batch = [v.to(model.device) for k, v in batch._asdict().items()]
                output = model(batch)
                loss = output.loss.mean()
                # loss = loss / args.num_grad_accum_steps
                loss.backward()

                for layer in range(num_layers):
                    model_layer = model.bert.encoder.layer[layer]
                
                    head_importance[domain][layer] += model_layer.attention.self.heads_mask.grad
                    ffn_importance[domain][layer] += model_layer.output.ffn_mask.grad[0]
                total_len += 1

            logger.info("***** Compute Average Importance *****")

            head_importance[domain] /= total_len
            ffn_importance[domain] /= total_len

            # Layerwise importance normalization
            if normalize_by_layer:
                exp = 2
                norm_by_layer = torch.pow(torch.pow(head_importance[domain], exp).sum(-1), 1/exp)
                head_importance[domain] /= norm_by_layer.unsqueeze(-1) + 1e-20

                norm = torch.pow(torch.pow(ffn_importance[domain], exp).sum(), 1/exp)
                ffn_importance[domain] /= norm.unsqueeze(-1) + 1e-20
    else:
        head_importance = torch.zeros(num_layers, num_heads).to(model.device)
        ffn_importance = torch.zeros(num_layers).to(model.device)

        model.eval()
        for batch in dataloaders:
            batch = [v.to(model.device) for k, v in batch._asdict().items()]
            output = model(batch)
            loss = output.loss.mean()
            loss.backward()

            for layer_idx in range(num_layers):
                model_layer = model.bert.encoder.layer[layer_idx]
                head_importance[layer_idx] += model_layer.attention.self.heads_mask.grad
                ffn_importance[layer_idx] += model_layer.output.ffn_mask.grad[0]

        head_importance /= len(dataloaders)
        ffn_importance /= len(dataloaders)

        # Layerwise importance normalization.
        if normalize_by_layer:
            exp = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exp).sum(-1), 1/exp)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

            norm = torch.pow(torch.pow(ffn_importance, exp).sum(), 1/exp)
            ffn_importance /= norm.unsqueeze(-1) + 1e-20

    return head_importance, ffn_importance


def get_head_sequence(percent_seq, min_attentionum_heads, num_heads=0, num_layers=0):
    # Compute the number of heads to prune on percent if needed.
    num_seq = []
    total_heads = num_heads * num_layers
    for percent in percent_seq:
        num_to_mask = int(total_heads * percent / 100)
        # Make sure we keep at least one head per layer.
        if min_attentionum_heads > 0:
            if num_to_mask > total_heads - min_attentionum_heads * num_layers:
                num_to_mask = total_heads - min_attentionum_heads * num_layers
                num_seq.append(num_to_mask)
                break
        num_seq.append(num_to_mask)

    # We'll incrementally prune layers and evaluate
    num_seq = sorted(num_seq)
    num_seq_ = num_seq[:]
    for idx in range(1, len(num_seq)):
        num_seq_[idx] = num_seq[idx] - num_seq[idx-1]

    # Verify that the total number of heads pruned stayed the same
    assert num_seq[-1] == sum(num_seq_)
    return num_seq_


def what_to_mask(head_importance, num_to_mask, min_attentionum_heads ,heads_to_mask=None, num_heads=0, num_layers=0, mask_reverse=False):
    # Sort heads by score
    if heads_to_mask is None:
        heads_to_mask = {}
    
    heads_and_score = [
            ((layer, head), head_importance[layer, head])
            for layer in range(num_layers)
            for head in range(num_heads)
        ]

    heads_and_score = sorted(heads_and_score, key=lambda x: x[1], reverse=mask_reverse)
    sorted_heads = [head_and_score[0] for head_and_score in heads_and_score]

    # Ensure we don't delete all heads in a layer
    if min_attentionum_heads:
        # Remove the top scoring head in each layer -> sorted_heads
        to_protect = {l: 0 for l in range(num_layers)}
        filtered_sorted_heads = []
        for layer, head in reversed(sorted_heads):
            if layer in to_protect:
                if to_protect[layer] < min_attentionum_heads:
                    to_protect[layer] += 1
                    continue
                else:
                    
                    to_protect.pop(layer)
            # position:0
            filtered_sorted_heads.insert(0, (layer, head))
        sorted_heads = filtered_sorted_heads
    #pdb.set_trace()
    # layer/heads that were already pruned
    # Prune the lowest scoring heads
    sorted_heads = [(layer, head) for (layer, head) in sorted_heads
        if layer not in heads_to_mask or head not in heads_to_mask[layer]
    ]

    # Update heads to prune
    for layer, head in sorted_heads[:num_to_mask]:
        if layer not in heads_to_mask:
            heads_to_mask[layer] = set()
        heads_to_mask[layer].add(head)

    return heads_to_mask

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj.item(), np.integer):
        return int(obj.item())
    elif isinstance(obj.item(), float):
        return obj.item()
    raise TypeError

def import_importance():
    head_importance, ffn_importance = {}, []
    num = 0
    with open('head_importance.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            data = list(line.strip('\n').split('\t'))
            head_importance[num] = {}
            for i, da in enumerate(data):
                head_importance[num][i] = da
            num += 1

    with open('ffn_importance.txt','r',encoding='utf-8') as f:
        data = list(f.readline().strip('\n').split('\t'))
        ffn_importance = []
        for i in data:
            ffn_importance.append(float(i))
    return head_importance, ffn_importance    
    
def save_importance(output_dir, head_importance, ffn_importance, domains=None):
    if domains is not None:
        for domain in domains:
            head_file = os.path.join(output_dir, f'{domain}_head_importance.txt')
            ffn_file = os.path.join(output_dir, f'{domain}_ffn_importance.txt')
            
            with open(head_file, 'w') as f:
                head_importance_list = head_importance[domain].cpu().tolist()
                for layer_score in head_importance_list:
                    for head_score in layer_score:
                        f.write(str(head_score)+"\t")
                    f.write("\n")

            with open(ffn_file, 'w') as f:
                ffn_importance_list = ffn_importance[domain].cpu().tolist()
                for score in ffn_importance_list:
                    f.write(str(score)+"\t")
    else:    
        head_file = os.path.join(output_dir, 'head_importance.txt')
        ffn_file = os.path.join(output_dir, 'ffn_importance.txt')

        with open(head_file, 'w') as f:
            head_importance_list = head_importance.cpu().tolist()
            for layer_score in head_importance_list:
                for head_score in layer_score:
                    f.write(str(head_score)+"\t")
                f.write("\n")

        with open(ffn_file, 'w', encoding='utf-8') as f:
            ffn_importance_list = ffn_importance.cpu().tolist()
            for score in ffn_importance_list:
                f.write(str(score)+"\t")

            
def dump_head_mask_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, default=set_default))

def dump_ffn_mask_file(path, data):
    with open(path,'w',encoding='utf-8')as f:
        for i in data:
            f.write(str(i)+'\n')

def load_head_mask_file(path):
    # heads_to_mask = {layer_id:[heads]}
    with open(path, "r", encoding='utf-8') as f:
        data = json.loads(f.readline().strip())
    heads_to_mask = {}
    for layer, heads in data.items():
        heads_to_mask[int(layer)] = set(heads)
    return heads_to_mask

def load_ffn_mask_file(path):
    # ffn_to_mask = [layer_id]
    ffn_to_mask = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            ffn_to_mask.append(int(line.strip()))
    ffn_to_mask = set(ffn_to_mask)
    return ffn_to_mask

def gen_random_head(heads_to_mask, num_heads, num_layers, preserve_layer_ratio=True):
    if preserve_layer_ratio:
        for layer, heads in heads_to_mask.items():
            heads_to_mask[layer] = set(np.random.choice(num_heads, len(heads)))
    else:
        total_num_heads = 0
        for layer, heads in heads_to_mask.items():
            total_num_heads += len(heads)
        heads_to_mask = {}
        random_heads = [i for i in range(num_heads * num_layers)]
        random.shuffle(random_heads)
    
        for head_id in random_heads[:total_num_heads]:
            if head_id // num_layers not in heads_to_mask:
                heads_to_mask[head_id // num_layers] = set()
            heads_to_mask[head_id // num_layers].add(head_id % num_heads)
    return heads_to_mask

def gen_random_ffn(ffn_to_mask, num_layers):
    random_ffn = [i for i in range(num_layers)]
    random.shuffle(random_ffn)
    ffn_to_mask = random_ffn[:len(ffn_to_mask)]
    return ffn_to_mask
