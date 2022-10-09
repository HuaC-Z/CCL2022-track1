#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import os, logging

import torch

from torch.utils.data import TensorDataset


from model.tokenization_funcun import FullTokenizer
from model.configuration_funcun import FuncunConfig
from model.modeling import FuncunCRASpell


import torch.distributed as dist


from common import init_logger, logger
from data.gen_model_data import read_test_file_2_features


# 加载数据
def load_features(file_path, is_file_path=True, is_test=False):
    if is_file_path:
        logger.info("start load file: {}".format(file_path))
        features = torch.load(file_path)
    else:
        logger.info("start load features")
        features = file_path

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)

    if not is_test:
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_noise_ids = torch.tensor([f.noise_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_lens, all_label_ids, all_noise_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_lens)

    if is_file_path:
        logger.info("finish load file: {}".format(file_path))

    else:
        logger.info("finish load features")

    return dataset


def collate_fn(batch, is_test=False):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    if not is_test:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_label_ids, all_noise_ids = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = \
            batch[0], batch[1], batch[2], batch[3]

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]

    if not is_test:
        all_label_ids = all_label_ids[:, :max_len]
        all_noise_ids = all_noise_ids[:, :max_len]

        return all_input_ids, all_attention_mask, all_token_type_ids, \
               all_label_ids, all_noise_ids

    return all_input_ids, all_attention_mask, all_token_type_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='model dir path')
    parser.add_argument("--env_pos", default="0", type=str,
                        help="cuda pos")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="cuda pos")
    parser.add_argument("--use_cuda", action="store_true",
                        help="use cuda or not")
    parser.add_argument("--save_path", default="../predict/CRASpell.lbl", type=str,
                        help="save predict file")
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.env_pos)
        if torch.cuda.is_available() and args.use_cuda:
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
            args.use_cuda = False

    args.device = device

    tokenizer = FullTokenizer(vocab_file=os.path.join(args.model_dir,
                                                      "vocab.txt"), do_lower_case=False)
    config = FuncunConfig.from_pretrained(os.path.join(args.model_dir,
                                                       "config.json"))

    config.alpha = 0
    init_logger("predict.log", log_file_level=logging.INFO)

    model = FuncunCRASpell.from_pretrained(args.model_dir,
                                           config=config)

    model.to(args.device)

    model.eval()

    fw = open(args.save_path,"w", encoding="utf-8")
    fw_sent_path = args.save_path.replace(".lbl", ".txt")
    fw_sent = open(fw_sent_path, "w", encoding="utf-8")
    test_features = read_test_file_2_features("data/yaclc-csc_test.src",
                                              tokenizer=tokenizer)
    test_dataset = load_features(test_features, is_test=True,
                                 is_file_path=False)

    print("test_dataset len: {}".format(len(test_dataset)))

    for i in range(0, len(test_dataset), 32):
        if i + 32 > len(test_dataset):
            epoch_dataset = test_dataset[i:]
        else:
            epoch_dataset = test_dataset[i: i + 32]

        batch = collate_fn(epoch_dataset, is_test=True)

        inputs = {}
        tag = {0: "input_ids", 1: "attention_mask", 2: "token_type_ids"}
        for idx, t in enumerate(batch):
            if idx not in tag: continue
            inputs[tag[idx]] = t.to(args.device)

        outputs = model.predict(**inputs)

        for offset, output in enumerate(outputs):
            fw.write("(YACLC-CSC-TEST-ID={})".format(
                str(i + offset + 1).zfill(4)))
            ori_sent = tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][offset][inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[1:-1]
            update_sent = tokenizer.convert_ids_to_tokens(
                output[inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[1:-1]

            assert (len(ori_sent) == len(update_sent))

            fw_sent.write("\nori_sent: {}\nupd_sent: {}\n".format(
                "".join(ori_sent), "".join(update_sent)))
            is_ok = True
            for j in range(len(ori_sent)):
                if len(ori_sent[j]) > 1 or len(update_sent[j]) > 1:
                    continue

                if ori_sent[j] != update_sent[j]:
                    fw.write(", {}, {}".format(j + 1, update_sent[j]))
                    is_ok = False

            if is_ok:
                fw.write(", 0")
            fw.write("\n")



