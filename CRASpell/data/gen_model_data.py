#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random, jieba
from common import get_ner_info, stringQ2B, logger
from common import is_chinese, is_chinese_string, has_chinese_string
from houxuanji import get_sim_yin_list, get_sim_xing_list, get_random_char_list
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


select_err_type_list = {
	1: 0.60,	# 音相似 60%
	2: 0.20,	# 形相似 20%
	3: 0.15,	# 颠倒 15%
	4: 0.05,	# 随机字 5%
}

noise_err_type_list = {
	1: 0.70,	# 音相似 70%
	2: 0.20,	# 形相似 20%
	3: 0.05,	# 颠倒 5%
	4: 0.05,	# 随机字 5%
}

def get_yinxiangsi_info(word, pos = 0, cdx = 0, ci_update_ratio = 0.5, is_nosie = False):

	logger.debug("word: {}, pos: {}, is_nosie: {}, yinxiangsi".format(word, pos, is_nosie))
	is_select = False; err_info = {}
	# 直接进行词替换的概率是0.5
	if random.random() < ci_update_ratio:
		
		select_words = get_sim_yin_list(word)
		if select_words:
			is_select = True

			err_word = random.choice(select_words)
			# 则记录信息, 错词，正词
			err_info["ori_word"] = word
			err_info["update_word"] = err_word
			err_info["start_pos"] = pos
			err_info["end_pos"] = pos + len(word)
			err_info["word_idx"] = cdx
			err_info["word_pos"] = pos 
			err_info["is_nosie"] = is_nosie
			# err_info_list.append(err_info)

	# 证明没有被选中，则进行到字的级别
	if not is_select:
		err_pos_list = [i for i in range(len(word))]
		random.shuffle(err_pos_list)
		for err_pos in err_pos_list:

			if not is_chinese(word[err_pos]): continue

			# logger.info("word: {}, err_pos: {}".format(word, err_pos))
			select_words = get_sim_yin_list(word, pos = err_pos)
			# logger.info("select_words: {}".format(select_words))
			if not select_words: continue
			is_select = True

			err_word = random.choice(select_words)

			# err_word = a_word[:err_pos] + err_char + a_word[err_pos + 1:]

			# 则记录信息, 错词，正词，
			err_info["ori_word"] = word
			err_info["update_word"] = err_word
			err_info["start_pos"] = pos
			err_info["end_pos"] = pos + len(word)
			err_info["word_idx"] = cdx
			err_info["word_pos"] = pos
			err_info["is_nosie"] = is_nosie
			break

	logger.debug("select_words: {}".format(select_words))
			
	return is_select, err_info

def get_xingxiangsi_info(word, pos = 0, cdx = 0, is_nosie = False):
	logger.debug("word: {}, pos: {}, is_nosie: {}, xingxiangsi".format(word, pos, is_nosie))
	is_select = False; err_info = {}

	err_pos_list = [i for i in range(len(word))]
	random.shuffle(err_pos_list)

	for err_pos in err_pos_list:
		if not is_chinese(word[err_pos]): continue
		select_chars = get_sim_xing_list(word[err_pos])

		if not select_chars: continue
		is_select = True

		err_char = random.choice(select_chars)

		# 则记录信息, 错词，正词，
		err_info["ori_word"] = word[err_pos]
		err_info["update_word"] = err_char
		err_info["start_pos"] = pos + err_pos
		err_info["end_pos"] = pos + err_pos + 1
		err_info["word_idx"] = cdx
		err_info["word_pos"] = pos
		err_info["is_nosie"] = is_nosie
		break

	
	return is_select, err_info

def get_diandao_info(word, pos = 0, cdx = 0, is_nosie = False):

	is_select = False; err_info = {}

	if len(word) == 1 or not is_chinese_string(word):
		return is_select, err_info

	cnt = 0
	while cnt < 5:
		err_pos_list = [i for i in range(len(word))]
		random.shuffle(err_pos_list)
		new_word = ""
		for a_pos in err_pos_list:
			new_word += word[a_pos]

		if new_word != word:
			break
		else:
			cnt += 1

	if cnt >= 5:
		return is_select, err_info

	is_select = True
	err_info["ori_word"] = word
	err_info["update_word"] = new_word
	err_info["start_pos"] = pos
	err_info["end_pos"] = pos + len(word)
	err_info["word_idx"] = cdx
	err_info["is_nosie"] = is_nosie
	err_info["word_pos"] = pos

	return is_select, err_info


def get_random_info(word, pos = 0, cdx = 0, is_nosie = False):
	
	logger.debug("word: {}, pos: {}, is_nosie: {}, random".format(word, pos, is_nosie))

	is_select = False; err_info = {}

	# 首先确定随机的替换位置
	err_pos_list = [i for i in range(len(word))]
	random.shuffle(err_pos_list)

	for err_pos in err_pos_list:
		if not is_chinese(word[err_pos]): continue

		select_chars = get_random_char_list()

		if not select_chars: continue

		random.shuffle(select_chars)
		for err_char in select_chars:
			if err_char != word[err_pos]:

				# 则记录信息, 错词，正词，
				err_info["ori_word"] = word[err_pos]
				err_info["update_word"] = err_char
				err_info["start_pos"] = pos + err_pos
				err_info["end_pos"] = pos + err_pos + 1
				err_info["word_idx"] = cdx
				err_info["is_nosie"] = is_nosie
				err_info["word_pos"] = pos
				is_select = True
				break

		if is_select: break
	
	return is_select, err_info



def get_new_sent(sent, err_info_list):

	err_info_list = sorted(err_info_list, key = lambda x: (x["start_pos"], -x["end_pos"]))

	logger.debug("sent: {}, err_info_list: {}".format(
		sent, err_info_list))
	offset = 0
	for err_info in err_info_list:
		word = err_info["update_word"]
		start_pos = err_info["start_pos"]; end_pos = err_info["end_pos"]
		new_sent = sent[:start_pos + offset] + word + sent[end_pos + offset: ]
		sent = new_sent

	return new_sent

def get_noise_info(sent, new_sent, err_info_list, ci_update_ratio = 0.05):
	
	is_select = False; err_info = {}	
	words = jieba.lcut(new_sent)
	err_pos_list = [i for i in range(len(err_info_list))]
	random.shuffle(err_pos_list)

	if len(err_pos_list) < 1:
		return is_select, err_info

	select_err_info = err_info_list[err_pos_list[0]]

	# 有5%的概率是词的音相似
	if random.random() < ci_update_ratio:
		exist_dict = {}
		for err_info in err_info_list:
			exist_dict[err_info["word_idx"]] = True

		select_cdx_list = []
		for i in range(-2, 3):
			if select_err_info["word_idx"] + i not in exist_dict and select_err_info["word_idx"] + i >= 0 and \
				select_err_info["word_idx"] + i < len(words) and \
				is_chinese_string(words[select_err_info["word_idx"] + i]):
				select_cdx_list.append(i)

		if select_cdx_list:
			random.shuffle(select_cdx_list)

			for select_cdx in select_cdx_list:

				cdx = select_err_info["word_idx"] + select_cdx
				if not is_chinese_string(words[cdx]): 
					continue
				if select_cdx < 0:

					pos = select_err_info["word_pos"]
					for i in range(-select_cdx):
						try:
							pos -= len(words[select_err_info["word_idx"] - i - 1])
						except Exception as e:
							print("words: {}, len: {}\ns_idx: {}, c_idx: {}".format(
								words, len(words), select_err_info["word_idx"], 
								select_cdx))
						

				elif select_cdx > 0:
					pos = select_err_info["word_pos"]
					for i in range(select_cdx):

						try:
							pos += len(words[select_err_info["word_idx"] + i + 1])
						except Exception as e:
							print("words: {}, len: {}\ns_idx: {}, c_idx: {}".format(
								words, len(words), select_err_info["word_idx"],
								select_cdx))

				is_select, info = get_yinxiangsi_info(words[cdx], 
					pos = pos, cdx = cdx)

				if is_select:
					break

	if not is_select:

		random_value = random.random()
		target_value = 0.0; select_err_type = -1
		for err_type in noise_err_type_list:

			if noise_err_type_list[err_type] + target_value >= random_value:
				select_err_type = err_type
				break

			target_value += noise_err_type_list[err_type]

		if select_err_type == -1:
			input("select_err_type == -1, has error")
			return is_select, err_info

		# 确定候选位置
		err_pos_list = [x for x in range(len(sent))]

		random.shuffle(err_pos_list)

		select_pos_list = []
		for i in range(select_err_info["start_pos"] - 5, select_err_info["end_pos"] + 5):
			if 0 <= i < len(sent) and sent[i] == new_sent[i] and is_chinese(sent[i]):
				select_pos_list.append(i)

		random.shuffle(select_pos_list)

		if select_pos_list:

			# 是音相似
			if select_err_type == 1:

				is_select, err_info = get_yinxiangsi_info(sent[select_pos_list[0]], 
					pos = select_pos_list[0], is_nosie = True, ci_update_ratio = 0.0)

			elif select_err_type == 2:
				is_select, err_info = get_xingxiangsi_info(sent[select_pos_list[0]],
					pos = select_pos_list[0], is_nosie = True)

			elif select_err_type == 3:
				pass

			elif select_err_type == 4:
				is_select, err_info = get_random_info(sent[select_pos_list[0]],
					pos = select_pos_list[0], is_nosie = True)

	logger.debug("noise select: {}, info: {}".format(is_select, err_info))
	return is_select, err_info

def get_train_infos(sent, name_pos):
	
	logger.debug("sent: {}, name_pos: {}".format(sent, name_pos))
	# 放入10%的正例子
	correct_sent_ratio = 0.1

	if random.random() <= correct_sent_ratio:
		info = {}
		info["ori_sent"] = sent
		info["trg_sent"] = sent
		info["noise_sent"] = sent
		info["items"] = []
		info["status"] = True
		return True, info

	# 每个词产生错误的概率是0.2
	words = jieba.lcut(sent)
	per_word_ratio = 0.2

	err_info_list = []
	pos = 0; pre_word_err = False
	for cdx, word in enumerate(words):
		is_name_word = False

		for ndx in range(len(name_pos)):
			name_start_pos, name_end_pos = name_pos[ndx]

			if not (name_start_pos >= pos + len(word) or name_end_pos <= pos):
				is_name_word = True
				break

			if name_start_pos >= pos + len(word):
				break


		if is_name_word:
			pos += len(word)
			pre_word_err = False
			continue

		# if not has_chinese_string(word):
		# 	pos += len(word)
		# 	pre_word_err = False
		# 	continue

		if not is_chinese_string(word):
			pos += len(word)
			pre_word_err = False
			continue

		random_value = random.random()
		if not pre_word_err and random_value < per_word_ratio:
			select_err_type = -1

			cur_random_value = random_value / per_word_ratio
			target_value = 0
			for err_type in select_err_type_list:

				if target_value + select_err_type_list[err_type] >= cur_random_value:
					select_err_type = err_type
					break
				target_value += select_err_type_list[err_type]

			if select_err_type == -1:
				input("select_err_type == -1, has error")
				pos += len(word)
				pre_word_err = False
				continue

			logger.debug("random_value: {}, select_err_type: {}".format(
				cur_random_value, select_err_type))

			# 错误类型是音相似时：
			if select_err_type == 1:

				is_select, err_info = get_yinxiangsi_info(word,
					pos = pos, cdx = cdx)

				if is_select:
					err_info["err_type"] = select_err_type
					err_info_list.append(err_info)
					pre_word_err = True
				else:
					pre_word_err = False

				pos += len(word)

			# 错误是形相似
			elif select_err_type == 2:
				
				is_select, err_info = get_xingxiangsi_info(word,
					pos = pos, cdx = cdx)

				if is_select:
					err_info["err_type"] = select_err_type
					err_info_list.append(err_info)
					pre_word_err = True
				else:
					pre_word_err = False

				pos += len(word)


			# 类型是颠倒
			elif select_err_type == 3:
				is_select, err_info = get_diandao_info(word, pos = pos, cdx = cdx)

				if is_select:
					err_info["err_type"] = select_err_type
					err_info_list.append(err_info)
					pre_word_err = True
				else:
					pre_word_err = False

				pos += len(word)

			# 类型是随机替换
			elif select_err_type == 4:
				is_select, err_info = get_random_info(word,
					pos = pos, cdx = cdx)

				if is_select:
					err_info["err_type"] = select_err_type
					err_info_list.append(err_info)
					pre_word_err = True
				else:
					pre_word_err = False

				pos += len(word)

			logger.debug("is_select: {}, info: {}".format(
				is_select, err_info))

		else:
			pre_word_err = False
			pos += len(word)


	if len(err_info_list) == 0:
		info = {}
		return False, info

	err_len = random.choice([i + 1 for i in range(min(2, max(len(sent) // 25, 1), 
		len(err_info_list)))])

	random.shuffle(err_info_list)
	err_info_list = err_info_list[:err_len]

	info = {}
	info["ori_sent"] = get_new_sent(sent, err_info_list)
	info["trg_sent"] = sent
	info["items"] = err_info_list
	# info["noise_sent"] = get_noise_sent(info["ori_sent"], err_info_list)

	is_select, err_info = get_noise_info(info["ori_sent"], sent, err_info_list)
	if is_select:
		info["noise_sent"] = get_new_sent(info["ori_sent"], [err_info])
		info["items"].append(err_info)
	else:
		info["noise_sent"] = info["ori_sent"]
	info["status"] = False

	return True, info

def get_err_info_list(sent1, sent2):

	err_info_list = []
	for i in range(len(sent1)):
		if sent1[i] != sent2[i]:
			info = {}
			info["ori_word"] = sent1[i]
			info["update_word"] = sent2[i]
			info["start_pos"] = i
			info["end_pos"] = i + 1
			info["word_idx"] = -1
			info["is_nosie"] = False

			err_info_list.append(info)

	return err_info_list


def read_file_2_features(file_path, tokenizer, is_only_sent = False, 
	split_str = " ||| ", re_ratio = 0.5, process_cnt = 5, desc = "train"):

	lines = []
	with open(file_path, "r", encoding = "utf-8") as f:
		for line in tqdm(f.readlines(), ncols = 80, desc = "R-line"):
			if not line or line.startswith("#") or line == "\n": 
				continue
			if line[-1] == "\n":
				line = line[:-1]
			lines.append(line)

	features = []

	if process_cnt > 1:
		with ProcessPoolExecutor(max_workers = process_cnt) as pool:
			p_cnt = len(lines) // process_cnt + 1
			thread_dict = {}
			for i in range(process_cnt):

				thread_dict[pool.submit(get_fetures, lines[i * p_cnt: (i + 1) * p_cnt], 
					tokenizer = tokenizer,
					is_only_sent = is_only_sent, 
					split_str = split_str, 
					re_ratio = re_ratio, idx = i)] = i

			for future in thread_dict:
				features.extend(future.result())
	else:
		features = get_fetures(lines, tokenizer = tokenizer, is_only_sent = is_only_sent,
			split_str = split_str, re_ratio = re_ratio)

	random.shuffle(features)
	print("train_cnt: {}".format(len(features)))
	return features	

def get_fetures(lines, tokenizer, is_only_sent = False, split_str = " ||| ",
	re_ratio = 0.5, idx = 0):

	infos = []
	for line in tqdm(lines, ncols = 80, desc = "H-line"):
		if is_only_sent:

			cnt = 0
			while cnt < 5:
				name_pos = get_ner_info(line)
				sta, info = get_train_infos(line, name_pos)
				if sta:
					infos.append(info)
					# input("info: {}\n\n".format(info))
					break
				else:
					cnt += 1
		else:
			sent1, sent2 = line.split(split_str)

			if len(sent1) != len(sent2) or len(sent1) > 510: continue

			if sent1 == sent2:

				info = {}
				info["ori_sent"] = sent1
				info["trg_sent"] = sent2
				info["noise_sent"] = sent1
				info["status"] = (sent1 == sent2)
				info["items"] = err_info_list.append(err_info)
				infos.append(info)

			else:
				err_info_list = get_err_info_list(sent2, sent1)

				cnt = 0
				while cnt < 3:
					sta, err_info = get_noise_info(sent1, sent2, 
						err_info_list, ci_update_ratio = -0.1)

					if sta:
						info = {}
						info["ori_sent"] = sent1
						info["trg_sent"] = sent2
						info["noise_sent"] = get_new_sent(sent1, [err_info])
						info["status"] = (sent1 == sent2)
						info["items"] = err_info_list.append(err_info)
						infos.append(info)
						break
					else:
						cnt += 1

			cur_ratio = re_ratio
			while random.random() < cur_ratio:

				cnt = 0
				while cnt < 3:
					name_pos = get_ner_info(sent2)
					sta, info = get_train_infos(sent2, name_pos)
					if sta:
						infos.append(info)
						break
					else:
						cnt += 1

				cur_ratio -= 1

	features = create_features(infos, tokenizer)

	return features


class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, input_len,
        label_ids = None, noise_ids = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label_ids = label_ids
        self.noise_ids = noise_ids


def get_ids(sent, tokenizer):

	tokens = []

	for char in sent:
		token = tokenizer.tokenize(char)
		if not token:
			tokens.append("[UNK]")
			continue
		tokens.append(token[0])

	tokens.insert(0, "[CLS]")
	tokens.append("[SEP]")

	ids = tokenizer.convert_tokens_to_ids(tokens)

	return ids


def create_features(infos, tokenizer, max_seq_len = 512):

	features = []

	for info in tqdm(infos, ncols = 80, desc = "G-features"):

		input_ids = get_ids(info["ori_sent"], tokenizer = tokenizer)

		attention_mask = [1] * len(input_ids)
		token_type_ids = [0] * len(input_ids)

		input_ids = input_ids[:max_seq_len]
		attention_mask = attention_mask[:max_seq_len]
		token_type_ids = attention_mask[:max_seq_len]
		input_len = len(input_ids)

		# while len(input_ids) < max_seq_len:
		for i in range(len(input_ids), max_seq_len):
			input_ids.append(0)
			attention_mask.append(0)
			token_type_ids.append(0)

		label_ids = None
		if info["trg_sent"]:
			label_ids = get_ids(info["trg_sent"], tokenizer = tokenizer)
			label_ids = label_ids[:max_seq_len]

			for i in range(len(label_ids), max_seq_len):
				label_ids.append(0)

		noise_ids = None
		if info["noise_sent"]:
			noise_ids = get_ids(info["noise_sent"], tokenizer = tokenizer)
			noise_ids = noise_ids[:max_seq_len]

			for i in range(len(noise_ids), max_seq_len):
				noise_ids.append(0)

		features.append(InputFeatures(input_ids = input_ids, 
			attention_mask = attention_mask,
			token_type_ids = token_type_ids,
			input_len = input_len,
			label_ids = label_ids,
			noise_ids = noise_ids))

	return features


def read_test_file_2_features(file_path, tokenizer):

	infos = []
	with open(file_path, "r", encoding = "utf-8") as f:
		for line in tqdm(f.readlines(), ncols = 80, desc = "R-test-line"):
			if not line or line.startswith("#") or line == "\n": 
				continue
			if line[-1] == "\n":
				line = line[:-1]

			line = line.split('\t')[1]

			info = {}
			info["ori_sent"] = line
			info["trg_sent"] = ""
			info["noise_sent"] = ""
			infos.append(info)

	features = create_features(infos, tokenizer)

	return features		




	












