import jieba
import random
import logging
import numpy as np
from transformers import BertTokenizer

from confusion_set import PinyinConfusionSet, StrokeConfusionSet
from common import load_idmap, load_labels, is_chinese_char


class PinyinTool:
    def __init__(self, py_dict_path, py_vocab_path):
        self.zi_pinyin = load_idmap(py_dict_path)
        vocab = load_labels(py_vocab_path)
        self.vocab = {'[PAD]': 0, '[UNK]': 1}
        self.vocab.update({k: v + 2 for k, v in vocab.items()})

    def get_pinyin_id(self, zi_unicode):
        py = self.zi_pinyin.get(zi_unicode, None)
        if py is None:
            return self.vocab['[UNK]']
        return self.vocab.get(py, self.vocab['[UNK]'])

    def get_tokenid_pyid(self, token_vocab):
        tokenid_pyid = {}
        for token in token_vocab:
            tokenid_pyid[token_vocab[token]] = self.get_pinyin_id(token)
        return tokenid_pyid


class Mask:
    def __init__(self, tokenizer: BertTokenizer,
                 max_mask,
                 masked_prob=0.15,
                 mask_masked_prob=0.8,
                 py_masked_prob=0,
                 jy_masked_prob=0,
                 sk_masked_prob=0,
                 random_masked_prob=0.1,
                 danzi_weight=None,
                 same_py_file='tools/confusions/same_pinyin.txt',
                 simi_py_file='tools/confusions/simi_pinyin.txt',
                 stroke_file='tools/confusions/same_stroke.txt'):
        self.max_mask = max_mask
        self.masked_prob = masked_prob
        self.mask_prob_thr = mask_masked_prob
        self.py_prob_thr = py_masked_prob + self.mask_prob_thr
        self.jy_prob_thr = jy_masked_prob + self.py_prob_thr
        self.sk_prob_thr = sk_masked_prob + self.jy_prob_thr
        self.random_prob_thr = random_masked_prob + self.sk_prob_thr
        if self.random_prob_thr > 1:
            logging.ERROR('掩码概率之和不能大于1！')

        self.danzi_weight = danzi_weight

        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.masked_id = tokenizer.vocab['[MASK]']
        self.invalid_ids = set(tokenizer.vocab[x] for x in ['[UNK]', '[CLS]', '[SEP]'])

        self.same_py_confu = PinyinConfusionSet(tokenizer, same_py_file)
        self.simi_py_confu = PinyinConfusionSet(tokenizer, simi_py_file)
        self.sk_confu = StrokeConfusionSet(tokenizer, stroke_file)

    def __call__(self, input_ids, label_ids=None):
        tokens = [self.tokenizer.decode(x) for x in input_ids]
        text = ''.join(tokens[1:-1])
        text = text.replace('[UNK]', '*')
        text_cut = jieba.lcut(text)
        danzi_pos = []
        for idx, word in enumerate(text_cut):
            if len(word) == 1 and is_chinese_char(word):
                pos = len(''.join(text_cut[:idx])) + 1
                danzi_pos.append(pos)

        if label_ids is None:
            label_ids = np.array([-100 for _ in range(len(input_ids))])
        cand_indices = [idx for idx, input_id in enumerate(input_ids) if input_id not in self.invalid_ids]
        num_to_mask = min(self.max_mask, max(1, int(round(len(cand_indices)) * self.masked_prob)))
        cand_weights = None
        if self.danzi_weight is not None:
            cand_weights = [1 if idx not in danzi_pos else self.danzi_weight for idx in cand_indices]
            cand_weights = np.array(cand_weights) / sum(cand_weights)
        if len(cand_indices) == 0:
            logging.info('数据里面没有汉字！')
            return
        new_cand_indices = np.random.choice(cand_indices, num_to_mask, replace=False, p=cand_weights)

        # random.shuffle(cand_indices)
        # for idx in cand_indices[:num_to_mask]:
        for idx in new_cand_indices:
            prob = random.random()
            masked_id = None
            if prob <= self.mask_prob_thr:
                masked_id = self.masked_id
            elif prob <= self.py_prob_thr:
                masked_id = self.same_py_confu.get_confusion_item_by_ids(input_ids[idx])
            elif prob <= self.jy_prob_thr:
                masked_id = self.simi_py_confu.get_confusion_item_by_ids(input_ids[idx])
            elif prob <= self.sk_prob_thr:
                masked_id = self.sk_confu.get_confusion_item_by_ids(input_ids[idx])
            elif prob <= self.random_prob_thr:
                masked_id = random.randint(0, self.vocab_size - 1)
            else:
                label_ids[idx] = input_ids[idx]
            if masked_id is not None:
                label_ids[idx] = input_ids[idx]
                input_ids[idx] = masked_id
