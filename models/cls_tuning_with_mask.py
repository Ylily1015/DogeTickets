# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from modules.modeling_bert_masked import GradBasedMaskedBertModel
import collections


Output = collections.namedtuple(
    "Output", 
    (
        "loss", 
        "prediction", 
        "label",
        "domain",
        "text_mask",
    )
)


class CLSTuningWithMask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = GradBasedMaskedBertModel(config)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label, domain = inputs
        
        hidden_states = self.bert(input_ids=text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        
        if len(label.size()) == 1:
            hidden_states = hidden_states[:, 0]
        logit = self.cls(hidden_states)

        if len(label.size()) > 1:
            batch_size, max_length = label.shape[0], label.shape[1]
            loss = F.cross_entropy(logit.view(batch_size * max_length,-1), label.view(batch_size * max_length,-1).squeeze(), reduction='none')
            
        else:
            loss = F.cross_entropy(logit, label, reduction='none')
        
        return Output(
            loss = loss, 
            prediction = logit.argmax(-1), 
            label = label,
            domain = domain,
            text_mask = text_mask,
        )


        
