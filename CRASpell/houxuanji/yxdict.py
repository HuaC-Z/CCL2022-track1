#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pypinyin import pinyin, Style
import pickle
import os, copy


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


ci_dict_path = "./houxuanji/dict_data/30wdict_utf8.txt"
shape_sim_data = "./houxuanji/dict_data/same_stroke.txt"
vocab_char_path = "./houxuanji/dict_data/vocab.txt"
yin_same_path = "./houxuanji/dict_data/same_pinyin.txt"
yin_simi_path = "./houxuanji/dict_data/simi_pinyin.txt"


special_char_list = ["的地得"]


class YXDict:

    def __init__(self):
        self.vocab_char_dict = self.get_vocab_char_dict()
        self.yin_ci_dict = self.get_yin_ci_dict()
        self.zi_zi_dict = self.get_zi_zi_dict()
        self.sim_yin_dict = self.get_sim_yin_dict()

    def get_sim_yin_dict(self):
        same_yin_dict = {}
        with open(yin_same_path, "r", encoding="utf-8") as f_same:
            for line in f_same.readlines():
                line = line.replace("\n", "").split('\t')
                if len(line) == 1:
                    continue
                if line[0] not in same_yin_dict:
                    same_yin_dict[line[0]] = set()
                o_chars = line[-1].split(" ")
                for each in o_chars:
                    if each in self.vocab_char_dict:
                        same_yin_dict[line[0]].add(each)
        with open(yin_simi_path, "r", encoding="utf-8") as f_simi:
            for line in f_simi.readlines():
                line = line.replace("\n", "").split('\t')
                if len(line) == 1:
                    continue
                if line[0] not in same_yin_dict:
                    same_yin_dict[line[0]] = set()
                o_chars = line[-1].split(" ")
                for each in o_chars:
                    if each in self.vocab_char_dict:
                        same_yin_dict[line[0]].add(each)
        return same_yin_dict

    def get_zi_zi_dict(self):
        sim_zi_zi_dict = {}

        with open(shape_sim_data, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split(',')
                if len(line) < 2:
                    continue
                for i, char in enumerate(line):
                    o_char = [line[j] for j in range(len(line)) if j != i]
                    if char not in sim_zi_zi_dict:
                        sim_zi_zi_dict[char] = set()
                    for each in o_char:
                        if each in self.vocab_char_dict:
                            sim_zi_zi_dict[char].add(each)

        return sim_zi_zi_dict

    # 获取特殊字的字典
    def get_special_zi_dict(self):
        special_zi_dict = {}

        for chars in special_char_list:
            for char in chars:
                if char not in special_zi_dict:
                    special_zi_dict[char] = set()
                for char_2 in chars:
                    special_zi_dict[char].add(char_2)

        return special_zi_dict

    def get_yin_ci_dict(self):
        yin_ci_dict = {}
        with open(ci_dict_path, 'r', encoding='utf-8') as f:
            for element in f.readlines():
                element = element.replace("\n", "")
                words = element.strip()

                py_list = self.get_words_py(words)

                for py in py_list:
                    if py not in yin_ci_dict:
                        yin_ci_dict[py] = set()
                    yin_ci_dict[py].add(words)

                    # 长度更大的 进行拆解加入
                    if len(words) > 2:
                        py_one_list = py.split("_")
                        for i in range(2, len(words), 1):
                            for j in range(0, len(words) - i + 1, 1):
                                a_py = "_".join(py_one_list[j: j + i])
                                a_word = words[j: j + i]
                                if a_py not in yin_ci_dict:
                                    yin_ci_dict[a_py] = set()
                                yin_ci_dict[a_py].add(a_word)

        return yin_ci_dict


    # 获取词表
    def get_vocab_char_dict(self):
        vocab_char_dict = {}

        with open(vocab_char_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if not line or line.startswith("#"): continue
                if line[-1] == "\n": line = line[:-1]
                line = line.strip()
                if len(line) != 1: continue
                if is_chinese(line) and line not in vocab_char_dict:
                    vocab_char_dict[line] = True

        return vocab_char_dict

    def dfs(self, w_py, pos, prefix, e_pos):

        if pos >= e_pos:
            return [prefix]

        ans = []
        for i in range(len(w_py[pos])):
            if prefix != "":
                ans.extend(self.dfs(w_py, pos + 1, "_".join([prefix, w_py[pos][i]]), e_pos))
            else:
                ans.extend(self.dfs(w_py, pos + 1, w_py[pos][i], e_pos))

        return ans

    def get_words_py(self, words):
        w_py = []
        for char in words:
            w_py.append(pinyin(char, style=Style.NORMAL, heteronym=True)[0])
        w_set = self.dfs(w_py, 0, "", len(w_py))
        return w_set

    def dump_yxdict(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_yxdict(cls, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


if __name__ == '__main__':
    yx = YXDict()

    while True:
        w_str = input("input words:").replace("\n", "").strip()
        py_list = yx.get_words_py(w_str)
        print("py list: {}".format(py_list))
        for py in py_list:
            print("pinyin: {}, words: {}".format(py, yx.yin_ci_dict.get(py, [])))

        print("\n\n\n")
