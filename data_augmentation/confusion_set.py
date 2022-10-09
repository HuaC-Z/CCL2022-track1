import random
import numpy as np
from collections import defaultdict


class ConfusionSet:
    def __init__(self, in_file):
        self.confusion = self._load_confusion(in_file)

    def _load_confusion(self, in_file):
        pass

    def is_confuse(self, token1, token2):
        confu1 = self.confusion.get(token1, [])
        confu2 = self.confusion.get(token2, [])
        if token1 in confu2 or token2 in confu1:
            return True
        return False

    def get_confusion_item(self, token):
        confu = self.confusion.get(token, None)
        if confu is None:
            return None
        return confu[random.randint(0, len(confu) - 1)]


class PinyinConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = {}
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()  # .decode('utf-8')
            tmps = line.split('\t')
            if len(tmps) != 2:
                continue
            key = tmps[0]
            values = list(set(tmps[1].split()))
            if len(values) > 0:
                confusion_datas[key] = values
        return confusion_datas


class StrokeConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file):
        confusion_datas = defaultdict(list)
        for line in open(in_file, encoding='utf-8'):
            line = line.strip()
            tmps = line.split(',')

            if len(tmps) < 2:
                continue
            values = list(set(tmps))
            for k in values:
                confusion_datas[k].extend(values)
        return confusion_datas


class YaclcConfusionSet(ConfusionSet):
    def __init__(self, in_file, temperature=2):
        super().__init__(in_file)
        self.tpt = temperature

    def _load_confusion(self, in_file):
        confusion_datas = {}
        with open(in_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                key = line[0]
                values = line[1].split()
                confusion_datas[key] = values
        return confusion_datas

    def get_choice_weights(self, length):
        pos_list = np.arange(length)
        weights = np.exp(-pos_list / self.tpt)
        weights = weights / np.sum(weights)
        return weights

    def get_confusion_item(self, token):
        confu = self.confusion.get(token, None)
        if confu is None:
            return None

        length = len(confu)
        if length == 1:
            return confu[0]
        choice_weights = self.get_choice_weights(length)

        return random.choices(confu, weights=choice_weights, k=1)[0]
