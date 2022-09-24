import re
import kenlm
import json
from eval_char_level import get_char_metrics
from eval_sent_level import get_sent_metrics

def detection():
    wfile = open("predict/detect.txt", "w", encoding="utf-8")

    with open("predict/cra.txt", "r", encoding="utf-8") as cra_rfile:
        lines_cra = cra_rfile.readlines()

    with open("predict/realise.txt", "r", encoding="utf-8") as realise_rfile:
        lines_realise = realise_rfile.readlines()

    with open("predict/roberta_01.txt", "r", encoding="utf-8") as roberta_01_rfile:
        lines_roberta_01 = roberta_01_rfile.readlines()

    for cra, realise, roberta_01 in zip(lines_cra, lines_realise, lines_roberta_01):
        ensembel = {}
        ID = cra.split(", ")[0]
        cra_list = cra.split(", ")[1:]
        realise_list = realise.split(", ")[1:]
        roberta_01_list = roberta_01.split(", ")[1:]

        if len(cra_list) != 1: 
            for idx, char in zip(cra_list[::2], cra_list[1::2]):
                ensembel[idx] = [char]

        if len(realise_list) != 1: 
            for idx, char in zip(realise_list[::2], realise_list[1::2]):
                if ensembel.get(idx) and char not in ensembel[idx]:
                    ensembel[idx].append(char)

        if len(roberta_01_list) != 1: 
            for idx, char in zip(roberta_01_list[::2], roberta_01_list[1::2]):
                if ensembel.get(idx):continue
                ensembel[idx] = [char]

        if ensembel == {}:
            wfile.write(f"{ID}, 0\n")
            continue 

        if len(ensembel.keys()) == 1:
            result = ', ' + list(ensembel.keys())[0] + ', #'
        else:
            result = ', '
            result += ', #, '.join(ensembel.keys())
            result += ', #'

        wfile.write(f"{ID}{result}\n")


def correction():
    n_gram =kenlm.LanguageModel('./n_gram/csc_char.bin')    

    with open("filter.txt", "r", encoding="utf-8") as filter_rfile:
        lines_cra = filter_rfile.readlines()
    filter_re = re.compile(f"{'|'.join(lines_cra)}")

    with open("candidate.json", "r", encoding="utf-8") as houxuanji_rfile:
        candidate_dict = json.load(houxuanji_rfile)
    
    with open ("./yaclc-csc_test.src", "r", encoding="utf-8") as src_rfile:
        src_lines = src_rfile.readlines()

    with open("predict/detect.txt", "r", encoding="utf-8") as pre_rfile:
        pre_lines = pre_rfile.readlines()
    
    wfile = open("predict/correct.txt", "w", encoding="utf-8")

    for src, pre in zip(src_lines, pre_lines):
        save_line = ""
        src = src.strip()
        pre = pre.strip()
        ID, src_text = src.split("\t")
        src_list = list(src_text)
        pre_list = pre.split(", ")[1:]
        save_line += ID
        if len(pre_list) == 1: 
            wfile.write(save_line + ", 0\n")
            continue
        for sentence_id, sentence_char in zip(pre_list[::2], pre_list[1::2]):
            sentence_id = int(sentence_id) - 1
            src_char = src_list[sentence_id]
            """引入过滤规则"""
            if re.search(filter_re, src_char): continue
            candidate_char = candidate_dict.get(src_char, "")
            candidate_tuple = []
            for cchar in candidate_char:
                tmp = src_list
                tmp[sentence_id] = cchar
                sentence = "".join(tmp)
                candidate_score = n_gram.score(sentence, bos = True, eos = True)
                candidate_tuple.append((candidate_score, cchar))
            candidate_tuple.sort(key=lambda x:x[0], reverse=True)
            if candidate_tuple == []: continue
            result_char = candidate_tuple[0][1]
            save_line += f", {sentence_id + 1}, {result_char}"
        wfile.write(f"{save_line}\n")

def evaluate(pred, gold, only_detection=False):
    char_metrics = get_char_metrics(pred, gold, only_detection)
    sent_metrics = get_sent_metrics(pred, gold, only_detection)
    return char_metrics, sent_metrics

# detection()
correction()
detection, correction = evaluate("predict/correct.txt", "yaclc-csc-test-5.1.lbl")







