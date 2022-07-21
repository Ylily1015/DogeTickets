# -*- coding: utf-8 -*-

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
import torch
from functools import partial
from conlleval import evaluate
import pdb
"""
Metric Facotry:
    Get metric function. [task-specific]
"""


def acc_and_f1(preds, labels, average="macro"):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"acc": acc, "f1": f1}


def acc(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"acc": acc}


def f1(preds, labels, average="macro"):
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"f1": f1}


def matthews(preds, labels):
    matthews_corr = matthews_corrcoef(y_true=labels, y_pred=preds)
    return {"matthews_corr": matthews_corr}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson_corr": pearson_corr, "spearman_corr": spearman_corr}


def decision_metric(preds, labels, domains, type, masks=None, reverse_label=None): 
    domain_list = {domain:{"preds": [], "labels": [], "masks": []} for domain in list(set(domains))} 
    
    for i, domain in enumerate(domains):  
        domain_list[domain]["preds"].append(preds[i])  
        domain_list[domain]["labels"].append(labels[i])  
        if masks is not None:
            domain_list[domain]["masks"].append(masks[i]) 
             
    acc_list = {}
    for domain, values in domain_list.items():
            
        if masks is not None:
            tensor_preds = torch.tensor(values["preds"])
            tensor_labels = torch.tensor(values["labels"])
            tensor_masks = torch.tensor(values["masks"])
            tmp_preds = torch.masked_select(tensor_preds, tensor_masks).numpy().tolist()
            tmp_labels = torch.masked_select(tensor_labels, tensor_masks).numpy().tolist()
            domain_preds, domain_labels = [],[]
            for pred, label in zip(tmp_preds, tmp_labels):
                domain_preds.append(reverse_label(pred))
                domain_labels.append(reverse_label(label))

            acc_list[domain] = evaluate(domain_labels, domain_preds)
            
            
        else:
            domain_preds = values["preds"]
            domain_labels = values["labels"]
            result = acc(domain_preds, domain_labels)
            acc_list[domain] = result["acc"]

    if type == "dev":
        avg_acc, var_acc = 0, 0

        for domain in domain_list:
            avg_acc += acc_list[domain] / len(domain_list)
        
        for domain in domain_list:
            var_acc += (acc_list[domain] - avg_acc) * (acc_list[domain] - avg_acc) / len(domain_list)

        return {"acc": avg_acc}
    else:
        return {"acc": acc_list}

 

METRIC_FN = {
    "mnli": decision_metric,
    "amazon": decision_metric,
    "ontonote":decision_metric,
}


def get_metric_fn(task_name):
    return METRIC_FN[task_name]
