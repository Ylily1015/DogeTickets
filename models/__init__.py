# -*- coding: utf-8 -*-

from transformers import (
    BertTokenizer,
    BertConfig,
)

from models.cls_tuning import CLSTuning
from models.cls_tuning_with_mask import CLSTuningWithMask



def get_model_class(model_type):
    if model_type == "cls_tuning":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuning
    elif model_type == "cls_tuning_with_mask":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = CLSTuningWithMask
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class