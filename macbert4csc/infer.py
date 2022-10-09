# -*- coding: utf-8 -*-

import sys
import torch
import argparse
from transformers import BertTokenizer
from loguru import logger
import operator
sys.path.append('../macbert4csc')

import os


from macbert4csc import MacBert4Csc
from defaults import _C as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(self, ckpt_path='output/macbert4csc/epoch=09-val_loss=0.08.ckpt',
                 vocab_path='output/macbert4csc/vocab.txt',
                 cfg_path='train_macbert4csc.yml'):
        logger.debug("device: {}".format(device))
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        cfg.merge_from_file(cfg_path)
        
        if 'macbert4csc' in cfg_path:
            self.model = MacBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                          cfg=cfg,
                                                          map_location=device,
                                                          tokenizer=self.tokenizer)
        else:
            raise ValueError("model not found.")
        self.model.to(device)
        self.model.eval()

    def predict(self, sentence_list):
        """
        文本纠错模型预测
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list)
        """
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list)
        if is_str:
            return corrected_texts[0]
        return corrected_texts

    def get_errors(corrected_text, origin_text):
        sub_details = []
        unk_tokens = [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤', '\t', '֍', '玕', '']
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                break
            if ori_char in unk_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def predict_with_error_detail(self, sentence_list):
        """
        文本纠错模型预测，结果带错误位置信息
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list), details(list)
        """
        details = []
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list)

        for corrected_text, text in zip(corrected_texts, sentence_list):
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            details.append(sub_details)
        if is_str:
            return corrected_texts[0], details[0]
        return corrected_texts, details

def make_lbl_file(src_path, save_path):

    with open(src_path, "r") as f:
        lines = f.readlines()
    wfile  = open(save_path, "w")

    for idx, line in enumerate(lines):
        src_sentence, trg_sentence = line.strip().split("\t")
        if src_sentence == trg_sentence:
            wfile.write(f"(YACLC-CSC-TEST-ID={str(idx+1).rjust(4,'0')}), 0\n")
        else:
            write_text = f"(YACLC-CSC-TEST-ID={str(idx+1).rjust(4,'0')})"
            for i, (src_char, trg_char) in enumerate(zip(src_sentence, trg_sentence)):
                if src_char != trg_char:
                    write_text += f", {i+1}, {trg_char}"
                wfile.write(f"{write_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--ckpt_path", default="macbert4csc/epoch=09-val_loss=0.08.ckpt", help="path to config file", type=str)
    parser.add_argument("--vocab_path", default="macbert4csc/vocab.txt", help="path to config file", type=str)
    parser.add_argument("--config_file", default="train_macbert4csc.yml", help="path to config file", type=str)
    parser.add_argument("--text_file", default='./yaclc-csc_test.src')
    parser.add_argument("--save_path", default='../predict/macbert4csc.lbl')
    args = parser.parse_args()
    m = Inference(args.ckpt_path, args.vocab_path, args.config_file)
    inputs = []
    with open(args.text_file, 'r', encoding='utf-8') as f:
        for line in f:
            inputs.append(line.strip())
    tmp_save_path = args.save_path.replace(".lbl", ".txt")
    save = open(tmp_save_path, "w", encoding="utf-8")

    outputs = m.predict(inputs)
    for input_s, output_s in zip(inputs, outputs):
        save.write(f"{input_s}\t{output_s}\n")

    make_lbl_file(tmp_save_path, args.save_path)

