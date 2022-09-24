#!/bin/bash
#_*_coding = utf-8

import torch, math
from torch import nn
from torchcrf import CRF
from tools.cross_loss import MyCrossEntropyLoss
from common import logger
import torch.nn.functional as F

def gelu(x):
	""" Original Implementation of the gelu activation function in Google Bert repo when initially created.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LSTM_CRF(nn.Module):
	"""docstring for LSTM_CRF"""
	def __init__(self, num_tag, input_size, hidden_size = 0, dropout = 0.1, 
		use_crf = True, use_lstm = True):
		super(LSTM_CRF, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.num_tag = num_tag
		self.use_crf = use_crf
		self.use_lstm = use_lstm
		# self.start_tag = num_tag
		# self.end_tag = num_tag + 1

		self.dropout = nn.Dropout(dropout)

		if use_lstm:
			self.lstm = nn.LSTM(self.input_size, self.hidden_size,
				batch_first = True, num_layers = 1,
				dropout = dropout, bidirectional = True)

			# 隐藏层转化为映射
			self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
			self.linear2 = nn.Linear(self.hidden_size, self.num_tag)
		
		else:
			self.linear3 = nn.Linear(self.input_size, self.num_tag)

		if self.use_crf:
			self.crf = CRF(num_tag, batch_first = True)

		else:
			self.cross_entry_loss = MyCrossEntropyLoss()
		# self.transitions = nn.Parameter(torch.Tensor(num_tag + 2, num_tag + 2))
		# nn.init.uniform_(self.transitions, -0.1, 0.1)
		# self.transitions.data[self.end_tag, :] = 100
		# self.transitions.data[:, self.start_tag] =  100
		self.use_cuda = torch.cuda.is_available()

	def rand_init_hidden(self, batch_size):

		h0 = torch.zeros(2, batch_size, self.hidden_size)
		c0 = torch.zeros(2, batch_size, self.hidden_size)

		if self.use_cuda:
			h0, c0 = h0.cuda(), c0.cuda()

		return (h0, c0)

	def forward(self, output, output_mask, label_ids):
		batch_size = output.size(0); seq_len = output.size(1)

		if self.use_lstm:
			output, _ = self.lstm(output, self.rand_init_hidden(batch_size))
			output = self.linear2(self.dropout(self.linear1(self.dropout(output))))

		else:
			output = self.linear3(self.dropout(output))

		output = gelu(output)

		# 使用crf
		if self.use_crf:
			
			
			loss = self.crf(output, label_ids, 
				mask = output_mask.byte(), reduction = "token_mean")

		# 不使用crf，则直接使用交叉熵
		else:
			total_loss, total_cnt = self.cross_entry_loss(
				output, label_ids, output_mask)

			loss = total_loss / total_cnt

		if self.use_lstm:
			del _

		# return loss
		return loss

	def decode(self, output, output_mask):
		batch_size = output.size(0); seq_len = output.size(1)
		# logger.info("use_lstm: {}, use_crf: {}".format(self.use_lstm, self.use_lstm))
		if self.use_lstm:
			output, _ = self.lstm(output, self.rand_init_hidden(batch_size))
			output = self.linear2(self.linear1(output))
		else:
			output = self.linear3(output)

		if self.use_lstm:
			del _

		output_2 = gelu(output)

		# 使用crf
		if self.use_crf:
			#output_2 = F.softmax(output_2, dim = 2)

			return self.crf.decode(output_2, output_mask.byte())

		# 不用crf，要转化为list
		else:
			res = []; ans = []
			output_2 = output_2.argmax(dim=2)
			output_2 = output_2[output_mask != 0]

			cur_idx = 0
			for idx, mask in enumerate(output_mask):
				cur_len = mask.sum().item()
				res.append(list(output_2[cur_idx: cur_idx + cur_len].cpu().numpy()))
				cur_idx += cur_len

			return res
