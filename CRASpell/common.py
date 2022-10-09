#_*_coding_*_ = UTF-8
from config import config
import os, random, torch
import numpy as np
from pyltp import Segmentor, Postagger, NamedEntityRecognizer
import logging
from pathlib import Path

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'

def is_chinese_string(string):
    """判断是否全为汉字"""
    # print(string, type(string), "@@@@")
    return all(is_chinese(c) for c in string)

def has_chinese_string(string):
    return any(is_chinese(c) for c in string)

def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])



# 分词模型
segment_words = None
# 词性标注
post_agger = None
# 人名识别
recognizer = None
def init_ltp_040():
    global segment_words, post_agger, recognizer

    # 初始化分词模型
    LTP_DATA_DIR = config["ltp_path"]

    # 分词模型初始化
    print("分词模型初始化开始")
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
    segment_words = Segmentor(cws_model_path)
    # segment_words.load(cws_model_path)
    print("分词模型初始化结束")

    # 词性标注模型初始化
    print("词性标注模型初始化开始")
    pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")
    post_agger = Postagger(pos_model_path)
    # post_agger.load(pos_model_path)
    print("词性标注模型初始化结束")

    # 命名实体识别模型初始化
    print("命名实体识别模型初始化开始")
    ner_model_path = os.path.join(LTP_DATA_DIR, "ner.model")
    recognizer = NamedEntityRecognizer(ner_model_path)
    # recognizer.load(ner_model_path)
    print("命名实体识别模型初始化结束")

def init_ltp_023():
    global segment_words, post_agger, recognizer

    # 初始化分词模型
    LTP_DATA_DIR = config["ltp_path"]

    # 分词模型初始化
    print("分词模型初始化开始")
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
    segment_words = Segmentor()
    segment_words.load(cws_model_path)
    print("分词模型初始化结束")

    # 词性标注模型初始化
    print("词性标注模型初始化开始")
    pos_model_path = os.path.join(LTP_DATA_DIR, "pos.model")
    post_agger = Postagger()
    post_agger.load(pos_model_path)
    print("词性标注模型初始化结束")

    # 命名实体识别模型初始化
    print("命名实体识别模型初始化开始")
    ner_model_path = os.path.join(LTP_DATA_DIR, "ner.model")
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)
    print("命名实体识别模型初始化结束")

# 初始化Ltp模型
def initLtp():
    print("start init ltp model")

    if config["ltp_version"] == "040":
        init_ltp_040()
    else:
        init_ltp_023()

    print("finish init ltp model")

name_2_tag = {
    "name": "Nh",
    "place": "Ns",
    "organ": "Ni"
}

# 提取相关的命名实体识别
def get_ner_info(sentence, ner_type = ["name"]):

    res = []; extract_type = [name_2_tag[e_type] for e_type in ner_type]
    if segment_words == None or post_agger == None or recognizer == None:
        return res

    # 先做分词
    words = segment_words.segment(sentence)

    # print("words: {}".format(" ".joinwords))
    # for word in words:
    #   print("word: {}".format(word))

    # 再做词性标注
    postags = post_agger.postag(words)
    # print("postags: {}".format(postags))

    # 最后做序列标注，得到结果
    netags = recognizer.recognize(words, postags)
    # print("netags: {}".format(netags))

    offset = 0; entity = ""; pos = 0
    for idx, tag in enumerate(netags):
        # print("idx: {}, tag: {}".format(idx, tag))
        while pos < len(sentence) and (sentence[pos] in [" ", "\u3000", "\t"]):
            pos += 1

        if tag.startswith("S-"):
            offset = pos

        if tag.startswith("E-") or tag.startswith("S-"):
            entity += words[idx]

            # cnt = 0
            # for char in entity:
            #   if is_chinese(char):
            #       cnt += 1

            # # 中文字符个数小于2，则不认为是人名
            # if cnt <= 1:
            #   entity = ""
            #   continue

            info = {}
            info["words"] = entity
            info["start_pos"] = offset
            info["end_pos"] = offset + len(entity)
            info["type"] = tag[2:]

            if tag[2:] in extract_type:
                res.append(info)
            entity = ""

        elif tag.startswith("B-"):
            entity = words[idx]
            offset = pos

        elif tag.startswith("I-"):
            entity += words[idx]

        pos += len(words[idx])

    return res