import os
def change_format(path, out_dir):
    with open(path, "r", encoding="utf-8") as rfile:
        lines = rfile.readlines()
    src_wfile = open(os.path.join(out_dir, "src.txt"), "w", encoding="utf-8")
    trg_wfile = open(os.path.join(out_dir, "trg.txt"), "w", encoding="utf-8")
    lbl_wfile = open(os.path.join(out_dir, "lbl.txt"), "w", encoding="utf-8")
    for i, line in enumerate(lines):
        line = line.strip()
        if line == "": continue
        src, trg = line.split("\t")
        src_wfile.write(f"{i+1}\t{src}\n")
        trg_wfile.write(f"{i+1}\t{trg}\n")
        lbl_line = ""
        no_error = True
        for id, trg_char in enumerate(trg):
            if trg_char != src[id]:
                no_error = False
                lbl_line += (str(id+1) + ", " + trg_char + ", ")
            if no_error:
                lbl_line += '0'
            lbl_line = lbl_line.strip(", ")
        lbl_wfile.write(lbl_line + "\n")

change_format("../data_augmentation/data/train.txt", "dataset/train")
change_format("../data_augmentation/data/dev.txt", "dataset/dev")