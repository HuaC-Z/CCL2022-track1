import jieba
import random
import argparse
from tqdm import tqdm
# import logging
# import pandas as pd

from confusion_set import PinyinConfusionSet, StrokeConfusionSet, YaclcConfusionSet
from utils import is_chinese_char


class yaclcAug:
    def __init__(self, yaclc_char_prob,
                 yaclc_word_prob,
                 replace_prob,
                 py_prob,
                 jy_prob,
                 sk_prob,
                 vocab_path='confusions/vocab.txt',
                 yaclc_char_file='confusions/confusion_char.txt',
                 yaclc_word_file='confusions/confusion_word.txt',
                 same_py_file='confusions/same_pinyin.txt',
                 simi_py_file='confusions/simi_pinyin.txt',
                 stroke_file='confusions/same_stroke.txt'):

        self.yaclc_word_prob = yaclc_word_prob
        self.yaclc_char_prob = yaclc_char_prob

        self.replace_prob = replace_prob
        self.py_prob_thr = py_prob
        self.jy_prob_thr = jy_prob + self.py_prob_thr
        self.sk_prob_thr = sk_prob + self.jy_prob_thr

        self.chinese_vocab = self.get_chinese_vocab(vocab_path)

        self.yaclc_char_confu = YaclcConfusionSet(yaclc_char_file)
        self.yaclc_word_confu = YaclcConfusionSet(yaclc_word_file)
        self.yaclc_char_keys = self.yaclc_char_confu.confusion.keys()
        self.yaclc_word_keys = self.yaclc_word_confu.confusion.keys()

        self.same_py_confu = PinyinConfusionSet(same_py_file)
        self.simi_py_confu = PinyinConfusionSet(simi_py_file)
        self.sk_confu = StrokeConfusionSet(stroke_file)

    def get_chinese_vocab(self, vocab_path):
        chinese_vocab = set()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if is_chinese_char(line):
                    chinese_vocab.add(line)
        return list(chinese_vocab)

    def word_replace(self, sent_list, word_cut, replace_indices):
        idx = 0
        for word in word_cut:
            if word in self.yaclc_word_keys:
                if random.random() <= self.yaclc_word_prob:
                    new_word = self.yaclc_word_confu.get_confusion_item(word)
                    if new_word is not None:
                        sent_list[idx: idx + len(word)] = list(new_word)
                        replace_indices[idx: idx + len(word)] = [1 for _ in range(len(word))]
            idx += len(word)

    def char_replace(self, sent_list, replace_indices):
        for idx in range(len(sent_list)):
            if replace_indices[idx] == 1:
                continue
            if sent_list[idx] in self.yaclc_char_keys and random.random() <= self.yaclc_char_prob:
                sent_list[idx] = self.yaclc_char_confu.get_confusion_item(sent_list[idx])
                replace_indices[idx] = 1

    def process(self, sent):
        length = len(sent)
        sent_list = list(sent)
        replace_indices = [0 for _ in range(length)]
        word_cut = jieba.lcut(sent)
        self.word_replace(sent_list, word_cut, replace_indices)
        self.char_replace(sent_list, replace_indices)

        for idx in range(length):
            if is_chinese_char(sent_list[idx]):
                if random.random() <= self.replace_prob:
                    prob = random.random()
                    if prob <= self.py_prob_thr:
                        new_char = self.same_py_confu.get_confusion_item(sent_list[idx])
                    elif prob <= self.jy_prob_thr:
                        new_char = self.simi_py_confu.get_confusion_item(sent_list[idx])
                    elif prob <= self.sk_prob_thr:
                        new_char = self.sk_confu.get_confusion_item(sent_list[idx])
                    else:
                        new_char = self.chinese_vocab[random.randint(0, len(self.chinese_vocab) - 1)]
                    if new_char is not None:
                        sent_list[idx] = new_char
                        replace_indices[idx] = 1

        if any(replace_indices):
            return ''.join(sent_list)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        help='Path to the source file',
                        required=True)
    parser.add_argument('-o', '--output_path',
                        help='Path to the target file',
                        required=True)
    args = parser.parse_args()

    aug = yaclcAug(yaclc_char_prob=0.1, yaclc_word_prob=0.01, replace_prob=0.01, py_prob=0.5, jy_prob=0.2, sk_prob=0.3)

    text_list = []
    with open(args.input_path, 'r', encoding='utf-8') as rfd:
        for line in rfd.readlines():
            line = line.strip().split('\t')[1].replace(' ', '')
            text_list.append(line)

    aug_text_list = []
    for text in tqdm(text_list, total=len(text_list)):
        new_text = aug.process(text)
        if new_text:
            aug_text_list.append((new_text, text))

    with open(args.output_path, 'w', encoding='utf-8') as wfd:
        for aug_text in aug_text_list:
            wfd.write(f'{(aug_text[0])}\t{aug_text[1]}\n')

