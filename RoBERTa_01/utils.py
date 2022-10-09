from torch.utils.data import DataLoader
from dataset import CSC_Dataset, Padding_in_batch
from eval_char_level import get_char_metrics
from eval_sent_level import get_sent_metrics, sent_metric_detect


# vocab_path = "./bert-base-chinese/vocab.txt"
# vocab_path = "./RoBERTa_zh_L12_PyTorch/vocab.txt"
vocab_path = './chinese_roberta_wwm_large_ext_L-24_H-1024_A-16_torch/vocab.txt'
vocab = []
with open(vocab_path, "r") as f:
    lines = f.readlines()
for line in lines:
    vocab.append(line.strip())


def init_dataloader(path, config, subset, tokenizer):

    sub_dataset = CSC_Dataset(path, config, subset)

    if subset == "train":
        is_shuffle = True
    else:
        is_shuffle = False

    collate_fn = Padding_in_batch(tokenizer.pad_token_id)

    data_loader = DataLoader(
        sub_dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )

    return data_loader


def csc_metrics(pred, gold):
    char_metrics = get_char_metrics(pred, gold)
    sent_metrics = get_sent_metrics(pred, gold)
    return char_metrics, sent_metrics


def get_best_score(best_score, best_epoch, epoch, *params):
    for para, key in zip(params, best_score.keys()):
        if para > best_score[key]:
            best_score[key] = para
            best_epoch[key] = epoch
    return best_score, best_epoch


# def save_decode_result_para(decode_pred, decode_pred_, data, path):
def save_decode_result_para(decode_pred, data, path):
    """
    decode_pred:模型预测结果
    data: 真实数据
    """
    f = open(path, "w")
    results = []
    for i, (pred_i, src) in enumerate(zip(decode_pred, data)):
        src_text = src['src_text']
        src_text_list = list(src_text)
        line, line_src = "", ""
        pred_i = pred_i[:len(src_text_list)]
        pred_i = pred_i[1:-1]
        line_src = ['0' for i in range(len(src_text))][1:-1]

        for idx, (i, j) in enumerate(zip(src_text[1:-1], src['trg_text'][1:-1])):
            if i != j:
                line_src[idx] = '1'

        for i, idx in enumerate(pred_i):
            tmp_char = ["0", "1"][idx]
            line += tmp_char
        # f.write("src:" + src['src_tex 
    f.close()


def save_decode_result_lbl(decode_pred, data, path):
    with open(path, "w") as fout:
        count = 0
        for pred_i, src in zip(decode_pred, data):
            src_text = src['src_text']
            src_text_list = list(src_text)
            tag_text = src['trg_text']
            if src_text == tag_text:
                count += 1
            item_id = src['id'].split("\t")[0]
            before, after = item_id.split("=")
            item_id = before + "=" + after[:-1].rjust(4, "0") + ")"
            line = item_id + ", "
            pred_i = pred_i[:len(src_text)]
            line_src = ['0' for i in range(len(src_text))]
            for idx, (i, j) in enumerate(zip(src_text, tag_text)):
                if i != j: 
                    line_src[idx] = '1'

            change_offset = []
            for i, idx in enumerate(pred_i):
                tmp_char = ["0", "1"][idx]
                if tmp_char == "1":
                    change_offset.append(i)

            
            for c_offset in change_offset:
                src_text_list[c_offset] = "想"

            pred_text = "".join(src_text_list)


            no_error = True
            for id, ele in enumerate(pred_i):
                if pred_text[id] != src_text[id]:
                    no_error = False
                    line += (str(id+1) + ", " + pred_text[id] + ", ")
            if no_error:
                line += '0'

            line = line.strip(", ")
            fout.write(line + "\n")
        print("count:",count)

def save_decode_result_lbl_01(decode_pred, data, path):
    with open(path, "w") as fout:
        for pred_i, src in zip(decode_pred, data):
            src_text = src['src_text']
            src_text_list = list(src_text)
            line = src['id'] + ", "
            pred_i = pred_i[:len(src_text)]

            change_offset = []
            for i, idx in enumerate(pred_i):
                tmp_char = ["0", "1"][idx]
                if tmp_char == "1":
                    change_offset.append(i)

            for c_offset in change_offset:
                src_text_list[c_offset] = "想"

            pred_text = "".join(src_text_list)

            no_error = True
            for id, ele in enumerate(pred_i):
                if pred_text[id] != src_text[id]:
                    no_error = False
                    line += (str(id+1) + ", " + pred_text[id] + ", ")
            if no_error:
                line += '0'

            line = line.strip(", ")
            fout.write(line + "\n")