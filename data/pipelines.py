# -*- coding: utf-8 -*-

import os
import csv
import json
import torch
import collections

from utils import Logger


logger = Logger()


class DataPipeline:
    def __init__(self, tokenizer, label_map, domain_map, max_length=None):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.domain_map = domain_map
        if max_length is None:
            self.max_length = tokenizer.model_max_length
        else:
            self.max_length = max_length

    @staticmethod
    def _truncate_pair(text_a_tokens, text_b_tokens, max_length):
        """Truncate a pair input in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(text_a_tokens) + len(text_b_tokens)
            if total_length <= max_length:
                break
            if len(text_a_tokens) > len(text_b_tokens):
                text_a_tokens.pop()
            else:
                text_b_tokens.pop()

    def build(self, examples, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return indices + [pad_idx] * pad_length

    def collate(self, batch):
        raise NotImplementedError()


class CombinedDataPipeline(DataPipeline):
    CombinedExample = collections.namedtuple(
        "CombinedExample", 
        (
            "text_indices", 
            "text_mask", 
            "text_segments", 
            "text_length", 
            "label",
            "domain"
        )
    )
    CombinedBatch = collections.namedtuple(
        "CombinedBatch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments", 
            "label",
            "domain"
        )
    )
    def __init__(self, tokenizer, label_map, domain_map, max_length=None):
        super().__init__(tokenizer, label_map, domain_map, max_length)

    @staticmethod
    def _truncate_pair(text_a_tokens, text_b_tokens, max_length):
        """Truncate a pair input in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(text_a_tokens) + len(text_b_tokens)
            if total_length <= max_length:
                break
            if len(text_a_tokens) > len(text_b_tokens):
                text_a_tokens.pop()
            else:
                text_b_tokens.pop()

    def build(self, examples, **kwargs):
        build_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            label = self.label_map(example.label)
            domain = self.domain_map(example.domain)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_b_tokens = None
            if example.text_b:
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3" for combined input.
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - 3)
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_a_tokens)
                text_tokens += text_b_tokens + [self.tokenizer.sep_token]
                text_segments += [1] * (len(text_b_tokens) + 1)
                text_length = len(text_tokens)
                text_mask = [1] * text_length
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                assert text_length <= self.max_length

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s (id = %d)" % (example.label, label))
                    logger.info("domain: %s (id = %d)" % (example.domain, domain))
                    
                build_examples.append(
                    CombinedDataPipeline.CombinedExample(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                        domain=domain
                    )
                )
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(text_a_tokens) > self.max_length - 2:
                    text_a_tokens = text_a_tokens[:(self.max_length - 2)]
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_tokens)
                text_length = len(text_tokens)
                text_mask = [1] * text_length
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s (id = %d)" % (example.label, label))
                    logger.info("domain: %s (id = %d)" % (example.domain, domain))

                build_examples.append(
                    CombinedDataPipeline.CombinedExample(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                        domain=domain
                    )
                )
        return build_examples

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return indices + [pad_idx] * pad_length

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([example.text_length for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_label = []
        batch_domain = []
        for example in batch:
            text_indices = self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id)
            batch_text_indices.append(text_indices)
            text_mask = self._pad(example.text_mask, max_length, 0)
            batch_text_mask.append(text_mask)
            text_segments = self._pad(example.text_segments, max_length, 0)
            batch_text_segments.append(text_segments)
            batch_label.append(example.label)
            batch_domain.append(example.domain)
            
        return CombinedDataPipeline.CombinedBatch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            label=torch.tensor(batch_label, dtype=torch.long),
            domain=torch.tensor(batch_domain, dtype=torch.long)
        )


class TagDataPipeline(DataPipeline):
    TagExample = collections.namedtuple(
        "TagExample", 
        (
            "text_indices", 
            "text_mask", 
            "text_segments", 
            "text_length", 
            "label",
            "domain"
        )
    )
    TagBatch = collections.namedtuple(
        "TagBatch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments", 
            "label",
            "domain"
        )
    )
    def __init__(self, tokenizer, label_map, domain_map, max_length=None):
        super().__init__(tokenizer, label_map, domain_map, max_length)

    @staticmethod
    def _truncate_pair(text_a_tokens, text_b_tokens, max_length):
        """Truncate a pair input in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(text_a_tokens) + len(text_b_tokens)
            if total_length <= max_length:
                break
            if len(text_a_tokens) > len(text_b_tokens):
                text_a_tokens.pop()
            else:
                text_b_tokens.pop()
                
    @staticmethod
    def tokens_label(text_a, label, tokenizer):
        text_a_split = [i.lower() for i in text_a.strip().split()]
        token_a_label = []
        for i, text in enumerate(text_a_split):
            tokens = tokenizer.tokenize(text)
            token_a_label.append(label[i])
            if tokens != [text]:
                if label[i][0]=="B":
                    follow_label = "I" + label[i][1:]
                else:
                    follow_label = label[i]
                token_a_label+= [follow_label] * (len(tokens) - 1)
                
        return token_a_label
    
    def build(self, examples, **kwargs):
        build_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))
            

            domain = self.domain_map(example.domain)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_a_label = self.tokens_label(example.text_a, example.label, self.tokenizer)
            label = []
            for each_label in text_a_label:
                label.append(self.label_map(each_label))
            text_b_tokens = None
            if example.text_b:
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3" for combined input.
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - 3)
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_a_tokens)
                text_tokens += text_b_tokens + [self.tokenizer.sep_token]
                text_segments += [1] * (len(text_b_tokens) + 1)
                text_length = len(text_tokens)
                text_mask = [1] * text_length
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                assert text_length <= self.max_length

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s (id = %d)" % (example.label, label))
                    logger.info("domain: %s (id = %d)" % (example.domain, domain))
                    
                build_examples.append(
                    CombinedDataPipeline.CombinedExample(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                        domain=domain
                    )
                )
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(text_a_tokens) > self.max_length - 2:
                    text_a_tokens = text_a_tokens[:(self.max_length - 2)]
                    label = label[:(self.max_length - 2)]
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                label = [0] + label + [0]
                
                text_segments = [0] * len(text_tokens)
                text_length = len(text_tokens)
                text_mask = [0] + [1] * (text_length-2) + [0]
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_mask: %s" % " ".join([str(x) for x in text_mask]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("text_length: %d" % text_length)
                    logger.info("label: %s" % " ".join([str(x) for x in label]))
                    logger.info("domain: %s (id = %d)" % (example.domain, domain))

                build_examples.append(
                    TagDataPipeline.TagExample(
                        text_indices=text_indices,
                        text_mask=text_mask,
                        text_segments=text_segments,
                        text_length=text_length,
                        label=label,
                        domain=domain
                    )
                )
        return build_examples

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return indices + [pad_idx] * pad_length

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([example.text_length for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_label = []
        batch_domain = []
        for example in batch:
            text_indices = self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id)
            batch_text_indices.append(text_indices)
            text_mask = self._pad(example.text_mask, max_length, 0)
            batch_text_mask.append(text_mask)
            text_segments = self._pad(example.text_segments, max_length, 0)
            batch_text_segments.append(text_segments)
            label = self._pad(example.label, max_length, 0)
            batch_label.append(label)
            batch_domain.append(example.domain)
            
        return TagDataPipeline.TagBatch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            label=torch.tensor(batch_label, dtype=torch.long),
            domain=torch.tensor(batch_domain, dtype=torch.long)
        )


    
    
