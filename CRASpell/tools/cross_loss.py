#_*_coding_*_ = UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCrossEntropyLoss(nn.Module):
	"""docstring for CrossEntryLoss"""
	# input: [batch_size, seq_len, size]
	# label: [batch_size, seq_len]
	# mask: [batch_size, seq_len]
	def __init__(self):
		super(MyCrossEntropyLoss, self).__init__()
		
	def forward(self, target, label, mask):
		if target.dim() > 2:
			target = target.contiguous().view(-1, target.size(2))

		label = label.contiguous().view(-1, 1)
		mask = mask.contiguous().view(-1, 1)
		# target = F.log(target, dim = 1)
		target = torch.log(target)
		loss = target.gather(1, label)
		tr_loss = (loss * mask).sum()
		total_cnt = (mask != 0).sum().item()

		return -tr_loss / total_cnt
