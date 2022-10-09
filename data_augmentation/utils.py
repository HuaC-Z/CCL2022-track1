import os
import yaml
import json
import requests
import logging
import pandas as pd
from pathlib import Path
from collections import OrderedDict


def load_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_labels(path):
    d = OrderedDict()
    index = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().split("\t")[0]
            d[token] = index
            index += 1
    return d


def load_idmap(path, sep='\t'):
    ans = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()  # .decode('utf8')
        tmps = line.split(sep)
        if len(tmps) != 2:
            continue
        ans[tmps[0]] = tmps[1]
    return ans


def load_table(path):
    if path.endswith('xlsx'):
        df = pd.read_excel(path)
    elif path.endswith('csv'):
        df = pd.read_csv(path, encoding='utf-8')
    else:
        df = pd.read_csv(path, encoding='utf-8', sep='\t')
    return df


def batch_yield(data_list, batch_size=256):
    length = len(data_list)
    i = 0
    while True:
        if i * batch_size >= length:
            break
        batch_data = data_list[i * batch_size:(i + 1) * batch_size]
        i += 1
        yield batch_data


def init_logger(log_file=None, log_file_level=logging.INFO):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        log_path = os.path.dirname(log_file)
        if len(log_path):
            os.makedirs(log_path, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def requests_post(sentences, ip='172.17.50.85', port='4000', app='/api/sentiment/classify', timeout=7):
    params = json.dumps(sentences)
    request_url = "http://{0}:{1}{2}".format(ip, port, app)
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers, timeout=timeout)
    return response.json()


def is_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def has_chinese_char(string):
    return any(is_chinese_char(c) for c in string)
