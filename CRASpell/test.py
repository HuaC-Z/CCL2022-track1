#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data.gen_model_data import read_file_2_features

from common import init_logger, logger, seed_everything
import logging
from model.tokenization_funcun import FullTokenizer

if __name__ == '__main__':

	seed_everything(49)
	init_logger("record.log", log_file_level = logging.INFO)
	# init_yxdict()

	tokenizer = FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)
	read_file_2_features("data/train.txt", tokenizer = tokenizer)

	# while True:
	# 	input_str = input("word: ").strip()
	# 	if input_str:
	# 		print("sim_info: {}\n\n".format(get_sim_words(input_str)))
	# read_file("sample.txt")
	
	# exist = {}
	# fw = open("data/train.txt", "w", encoding = "utf-8")

	# file_path_list = ["data/nn_train.txt", "data/wang.train.ccl22.para"]
	
	# for file_path in file_path_list:
	# 	with open(file_path, "r", encoding = "utf-8") as f:
	# 		for line in f.readlines():
	# 			if not line: continue
	# 			if line.startswith("#"): continue
	# 			if line not in exist:
	# 				exist[line] = True
	# 				fw.write(line)
