# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

import collections
import pdb

Output = collections.namedtuple(
    "Output", 
    (
        'loss', 
        'prediction', 
        'label',
    )
)


class CLSTuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label, _ = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        hidden_states = hidden_states[:, 0]

        logit = self.cls(hidden_states)
        #bertpdb.set_trace()
        loss = F.cross_entropy(logit, label, reduction='none')
        
        return Output(
            loss=loss, 
            prediction=logit.argmax(-1), 
            label=label,
        )


        
