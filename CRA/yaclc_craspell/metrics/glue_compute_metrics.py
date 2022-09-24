# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import sys
import logging

logger = logging.getLogger(__name__)
# try:
#     # from scipy.stats import pearsonr, spearmanr
#     # from sklearn.metrics import matthews_corrcoef, f1_score
#     from sklearn.metrics import f1_score
#     _has_sklearn = True
# except (AttributeError, ImportError) as e:
#     logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
#     _has_sklearn = False

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


# def acc_and_f1(preds, labels, average = 'binary'):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds, average = average)
#     return {
#         "acc": acc,
#         "f1": f1,
#         #"acc_and_f1": (acc + f1) / 2,
#     }

# def pearson_and_spearman(preds, labels):
#     pearson_corr = pearsonr(preds, labels)[0]
#     spearman_corr = spearmanr(preds, labels)[0]
#     return {
#         "pearson": pearson_corr,
#         "spearmanr": spearman_corr,
#         "corr": (pearson_corr + spearman_corr) / 2,
#     }

def calc_f1(preds, labels):

    # print("preds: {}\nlabels: {}\n==: {}\nres: {}".format(
        # preds, labels, preds == labels, (preds == labels) * (labels != 0)))

    TP = ((preds == labels) * (labels != 0)).sum()

    # input("TP: {}".format(TP))

    if (preds != 0).sum() == 0:
        if (labels != 0).sum() == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = TP / ((preds != 0).sum())

    if (labels != 0).sum() == 0:
        if (preds != 0).sum() == 0:
            recall = 1.0
        else:
            recall = 0.0

    else:
        recall = TP / ((labels != 0).sum())

    if abs(precision + recall) <= 1e-6:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1 
    }



def compute_metrics(task_name, preds, labels, metr = "macro"):
    assert len(preds) == len(labels)
    # print("task_name: ", task_name)
    if task_name == "ner":
        return calc_f1(preds, labels, metr)
    elif task_name == "csc":
        return calc_f1(preds, labels)
    else:
        raise KeyError(task_name)
