#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import os, time, logging
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model.tokenization_funcun import FullTokenizer
from model.configuration_funcun import FuncunConfig
from model.modeling import FuncunCRASpell

from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
import torch.distributed as dist
# from torch.nn import CrossEntropyLoss, MSELoss
import dill

from common import init_logger, seed_everything, logger
from metrics.glue_compute_metrics import compute_metrics
from data.gen_model_data import read_file_2_features, read_test_file_2_features


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def collate_fn(batch, is_test=False):
    """
	batch should be a list of (sequence, target, length) tuples...
	Returns a padded tensor of sequences sorted from longest to shortest,
	"""
    if not is_test:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_label_ids, all_noise_ids = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = \
            batch[0], batch[1], batch[2], batch[3]

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]

    if not is_test:
        all_label_ids = all_label_ids[:, :max_len]
        all_noise_ids = all_noise_ids[:, :max_len]

        return all_input_ids, all_attention_mask, all_token_type_ids, \
               all_label_ids, all_noise_ids

    return all_input_ids, all_attention_mask, all_token_type_ids


def train(args, model, tokenizer):
    """ Train the model """
    cached_file_dir = args.cached_file_dir
    if args.local_rank == -1 or args.local_rank == args.calc_rank:

        train_features = read_file_2_features(args.train_data_path, tokenizer=tokenizer,
                                              is_only_sent=False, split_str="\t", re_ratio=0.5, desc="train")

        valid_features = read_file_2_features(args.valid_data_path, tokenizer=tokenizer,
                                              is_only_sent=False, split_str="\t", re_ratio=1.0, desc="valid")

        if not os.path.exists(cached_file_dir):
            os.makedirs(cached_file_dir)

        if args.local_rank != -1:
            logger.info("start save train")
            torch.save(train_features, os.path.join(cached_file_dir, "cached_train"))
            logger.info("finishi save train")

        logger.info("start save valid")
        torch.save(valid_features, os.path.join(cached_file_dir, "cached_valid"))
        logger.info("finish save valid")

    # logger.info("local_rank333: {}".format(args.local_rank))
    # if args.local_rank != -1 and args.local_rank != args.calc_rank:
    if args.local_rank != -1:
        dist.barrier()

    valid_dataset = load_features(os.path.join(cached_file_dir, "cached_valid"))
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  batch_size=args.per_gpu_eval_batch_size, collate_fn=collate_fn)

    epoch_sample = 270000

    num_training_steps = epoch_sample * args.num_train_epochs // args.per_gpu_train_batch_size

    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    if args.retrain:

        if "optim" in os.listdir(args.model_dir):
            optim_path = os.path.join(args.model_dir, "optim")
            optim_info = torch.load(optim_path, pickle_module=dill)
            logger.info("optim_info: step: {}".format(optim_info["step"]))
            ever_lr = optim_info["optimizer"].param_groups[0]["lr"]
            logger.info("optime lr: {}".format(ever_lr))

            optimizer = AdamW(params=optimizer_grouped_parameters,
                              lr=(ever_lr + args.learning_rate) * 2 / 3,
                              eps=args.adam_epsilon)

            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_training_steps)
    else:
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_training_steps)

    train_batch_size = args.per_gpu_train_batch_size
    eval_batch_size = args.per_gpu_eval_batch_size

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", epoch_sample)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", train_batch_size)
    logger.info("  Total optimization steps = %d", num_training_steps)

    tr_loss = 0.0
    best_f1 = 0;
    no_add = 0;
    total_num = 0;
    global_step = 0
    model.zero_grad()

    t1 = time.time()

    # logger.info("local_rank000: {}".format(args.local_rank))
    # if args.local_rank != -1 and args.local_rank != args.calc_rank:
    # dist.barrier()

    # loss_fn = MSELoss()

    best_f1 = 0.0;
    best_model = None;
    no_add = 0
    for _ in range(int(args.num_train_epochs)):

        if _ > 0 and (args.local_rank == -1 or args.local_rank == args.calc_rank):
            train_features = read_file_2_features(args.train_data_path, tokenizer=tokenizer,
                                                  is_only_sent=False, split_str="\t", re_ratio=0.5, desc="train")

            if args.local_rank != -1:
                logger.info("start save train")
                torch.save(train_features, os.path.join(cached_file_dir, "cached_train"))
                logger.info("finish save train")

        # logger.info("local_rank111: {}".format(args.local_rank))
        # if args.local_rank != -1 and args.local_rank != args.calc_rank:
        if args.local_rank != -1:
            dist.barrier()

        epoch_num = 0;
        epoch_loss = 0

        # logger.info("local_rank222: {}".format(args.local_rank))
        # train_dataset = load_features(os.path.join(cached_file_dir, "cached_train"))

        if args.local_rank == -1:
            train_dataset = load_features(train_features, is_file_path=False)
        else:
            train_dataset = load_features(os.path.join(cached_file_dir, "cached_train"))

        if args.local_rank != -1:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                          batch_size=train_batch_size, collate_fn=collate_fn)
        else:
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                          batch_size=train_batch_size, collate_fn=collate_fn)

        pbar = ProgressBar(n_total=len(train_dataloader),
                           desc='ep:{}'.format(int(_ + 1)))

        if args.local_rank != -1:
            train_sampler.set_epoch(_)
        model.train()
        for step, batch in enumerate(train_dataloader):
            # if step >= 10: break
            inputs = {}
            tag = {0: "input_ids", 1: "attention_mask", 2: "token_type_ids",
                   3: "label_ids", 4: "noise_ids"}
            for idx, t in enumerate(batch):
                if idx not in tag: continue
                inputs[tag[idx]] = t.to(args.device)

            # seq_label_loss, class_loss = model(**inputs)

            # batch_loss = seq_label_loss * 0.5 + class_loss * 0.6
            loss, __, kl_loss, cross_loss = model(**inputs)

            logger.debug("\nloss: {}, kl_loss: {}, cross_loss: {}".format(
                loss, kl_loss, cross_loss))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            step_num = (inputs["attention_mask"] != 0).sum().item()
            epoch_num += step_num
            epoch_loss += loss.item() * step_num
            # global_step += 1; epoch_step += 1

            if args.local_rank != -1:
                batch_loss = reduce_mean(loss, dist.get_world_size())

            pbar(step, {'loss': loss.item(), "kl_loss": kl_loss.item(),
                        "cross_loss": cross_loss.item(),
                        "ep_ag_ls:": epoch_loss / epoch_num})

        del train_dataloader, train_sampler, train_dataset

        logger.debug("epoch_num: {}, epoch_loss: {}".format(epoch_loss, epoch_num))
        logger.debug("\nepoch: {}, epoch_avg_loss: {}".format(int(_ + 1),
                                                              epoch_loss / epoch_num))

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

        # 评估一下
        if args.local_rank == -1 or args.local_rank == args.calc_rank:

            result = evaluate(args, model, valid_dataloader,
                              dataset_size=len(valid_dataset),
                              batch_size=args.per_gpu_eval_batch_size, data_type="valid")

            if result["f1"] > best_f1:

                opt_info = {}
                opt_info["step"] = global_step
                opt_info["num_training_steps"] = num_training_steps
                opt_info["num_warmup_steps"] = args.warmup_steps
                opt_info["optimizer"] = optimizer
                opt_info["scheduler"] = scheduler

                output_dir = os.path.join(args.out_dir, 'checkpoint-%02d' % (int(_ + 1)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if args.local_rank == -1:
                    model.save_pretrained(output_dir, optim=opt_info)
                else:
                    model.module.save_pretrained(output_dir, optim=opt_info)
                logger.info("Saving model checkpoint to %s", output_dir)
                no_add = 0

                best_model = model
                best_f1 = result["f1"]

                # 得到test输出
                fw = open("yaclc-csc-test_{}.lbl".format(str(_ + 1).zfill(2)),
                          "w", encoding="utf-8")
                fw_sent = open("yaclc-sent_{}.txt".format(str(_ + 1).zfill(2)),
                               "w", encoding="utf-8")
                test_features = read_test_file_2_features("data/yaclc-csc_test.src",
                                                          tokenizer=tokenizer)
                test_dataset = load_features(test_features,
                                             is_test=True, is_file_path=False)

                for i in range(0, len(test_dataset), args.per_gpu_eval_batch_size):
                    if i + args.per_gpu_eval_batch_size > len(test_dataset):
                        epoch_dataset = test_dataset[i:]
                    else:
                        epoch_dataset = test_dataset[i: i + args.per_gpu_eval_batch_size]

                    batch = collate_fn(epoch_dataset, is_test=True)

                    inputs = {}
                    tag = {0: "input_ids", 1: "attention_mask", 2: "token_type_ids"}
                    for idx, t in enumerate(batch):
                        if idx not in tag: continue
                        inputs[tag[idx]] = t.to(args.device)

                    if args.local_rank == -1:
                        outputs = best_model.predict(**inputs)
                    elif args.local_rank == args.calc_rank:
                        outputs = best_model.module.predict(**inputs)

                    for offset, output in enumerate(outputs):
                        fw.write("(YACLC-CSC-TEST-ID={})".format(
                            str(i + offset + 1).zfill(4)))
                        ori_sent = tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][offset][inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[
                                   1:-1]
                        update_sent = tokenizer.convert_ids_to_tokens(
                            output[inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[1:-1]

                        assert (len(ori_sent) == len(update_sent))

                        fw_sent.write("\nori_sent: {}\nupd_sent: {}\n".format(
                            "".join(ori_sent), "".join(update_sent)))
                        is_ok = True
                        for j in range(len(ori_sent)):
                            if len(ori_sent[j]) > 1 or len(update_sent[j]) > 1:
                                continue

                            if ori_sent[j] != update_sent[j]:
                                fw.write(", {}, {}".format(j + 1, update_sent[j]))
                                is_ok = False

                        if is_ok:
                            fw.write(", 0")
                        fw.write("\n")

            else:
                no_add += 1
                if no_add >= args.no_add:
                    logger.info("early stop, epoch: {}".format(int(_ + 1)))
                    break

        # elif args.local_rank == args.calc_rank:

        # 	result = evaluate(args, model, valid_dataloader,
        # 		batch_size = args.per_gpu_eval_batch_size, data_type = "valid")

        # 	if result["f1"] > best_f1:
        # 		logger.info("Saving model checkpoint to %s", output_dir)
        # 		opt_info = {}
        # 		opt_info["step"] = global_step
        # 		opt_info["num_training_steps"] = num_training_steps
        # 		opt_info["num_warmup_steps"] = args.warmup_steps
        # 		opt_info["optimizer"] = optimizer
        # 		opt_info["scheduler"] = scheduler

        # 		output_dir = os.path.join(args.out_dir, 'checkpoint-%02d'%(int(_ + 1)))
        # 		if not os.path.exists(output_dir):
        # 			os.makedirs(output_dir)
        # 		model.save_pretrained(output_dir, optim = opt_info)
        # 		no_add = 0
        # 	else:
        # 		no_add += 1
        # 		if no_add >= args.no_add:
        # 			logger.info("early stop, epoch: {}".format(int(_ + 1)))
        # 			break

        # opt_info = {}
        # opt_info["step"] = global_step
        # opt_info["num_training_steps"] = num_training_steps
        # opt_info["num_warmup_steps"] = args.warmup_steps
        # opt_info["optimizer"] = optimizer
        # opt_info["scheduler"] = scheduler

        # out_dir = os.path.join(args.out_dir, 'checkpoint-%02d'%(int(_ + 1)))
        # if not os.path.exists(out_dir):
        # 	os.makedirs(out_dir)
        # model.module.save_pretrained(out_dir, optim = opt_info)
        # # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        # logger.info("Saving model checkpoint to %s", out_dir)
        # result = evaluate(args, model, valid_dataloader,
        # 	batch_size = args.per_gpu_eval_batch_size, data_type = "valid")

        if args.local_rank != -1:
            dist.barrier()

    if args.local_rank == -1 or args.local_rank == args.calc_rank:
        # 得到test输出
        fw = open("yaclc-csc-test_final.lbl",
                  "w", encoding="utf-8")
        fw_sent = open("yaclc-sent_final.txt", "w", encoding="utf-8")
        test_features = read_test_file_2_features("data/yaclc-csc_test.src",
                                                  tokenizer=tokenizer)
        test_dataset = load_features(test_features, is_test=True,
                                     is_file_path=False)

        for i in range(0, len(test_dataset), args.per_gpu_eval_batch_size):
            if i + args.per_gpu_eval_batch_size > len(test_dataset):
                epoch_dataset = test_dataset[i:]
            else:
                epoch_dataset = test_dataset[i: i + args.per_gpu_eval_batch_size]

            batch = collate_fn(epoch_dataset, is_test=True)

            inputs = {}
            tag = {0: "input_ids", 1: "attention_mask", 2: "token_type_ids"}
            for idx, t in enumerate(batch):
                if idx not in tag: continue
                inputs[tag[idx]] = t.to(args.device)

            if best_model == None:
                best_model = model
            if args.local_rank == -1:
                outputs = best_model.predict(**inputs)
            elif args.local_rank == args.calc_rank:
                outputs = best_model.module.predict(**inputs)

            for offset, output in enumerate(outputs):
                fw.write("(YACLC-CSC-TEST-ID={})".format(
                    str(i + offset + 1).zfill(4)))
                ori_sent = tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][offset][inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[1:-1]
                update_sent = tokenizer.convert_ids_to_tokens(
                    output[inputs["attention_mask"][offset] != 0].cpu().numpy().tolist())[1:-1]

                assert (len(ori_sent) == len(update_sent))

                fw_sent("\nori_sent: {}\nupd_sent: {}\n".format(
                    "".join(ori_sent), "".join(update_sent)))
                is_ok = True
                for j in range(len(ori_sent)):
                    if len(ori_sent[j]) > 1 or len(update_sent[j]) > 1:
                        continue

                    if ori_sent[j] != update_sent[j]:
                        fw.write(", {}, {}".format(j + 1, update_sent[j]))
                        is_ok = False

                if is_ok:
                    fw.write(", 0")
                fw.write("\n")

    if args.local_rank != -1:
        dist.barrier()
    logger.info("finish, cost time: {}".format(int((time.time() - t1) * 1000) / 1000))


def evaluate(args, model, valid_dataloader, batch_size, dataset_size, prefix="", data_type="valid"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    results = {}

    # Eval!
    logger.info("***** Running Evaluation*****")
    logger.info("  Num examples = %d", dataset_size)
    logger.info("  Batch size = %d", batch_size)

    pbar = ProgressBar(n_total=len(valid_dataloader), desc="valid")
    # valid_sampler.set_epoch(0)
    epoch_num = 0;
    epoch_loss = 0
    predict_label = [];
    real_label = []
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            inputs = {}
            tag = {0: "input_ids", 1: "attention_mask", 2: "token_type_ids",
                   3: "label_ids", 4: "noise_ids"}
            for idx, t in enumerate(batch):
                if idx not in tag: continue
                inputs[tag[idx]] = t.to(args.device)

            loss, output, kl_loss, cross_loss = model(**inputs)

            step_num = (inputs["attention_mask"] != 0).sum().item()
            epoch_num += step_num
            epoch_loss += loss.item() * step_num

        pbar(step, {'loss': loss.item(), "kl_loss": kl_loss.item(),
                    "cross_loss": cross_loss.item(),
                    "ep_ag_ls:": epoch_loss / epoch_num})

        real_batch_label = (inputs["input_ids"] == inputs["label_ids"])
        real_batch_label = real_batch_label[inputs["attention_mask"] != 0]
        real_batch_label = (1 - real_batch_label.int()).int().cpu().numpy().tolist()

        predict_batch_label_1 = 1 - (output == inputs["input_ids"]).int()
        predict_batch_label_2 = 1 - (output == inputs["label_ids"]).int()

        predict_batch_label = (predict_batch_label_1 + predict_batch_label_2) * \
                              predict_batch_label_1

        predict_batch_label = predict_batch_label[inputs["attention_mask"] != 0].int().cpu().numpy().tolist()

        predict_label.extend(predict_batch_label)
        real_label.extend(real_batch_label)

    logger.info("\nepoch_avg_loss: {}".format(epoch_loss / epoch_num))

    # predict_label = np.array(predict_label); real_label = np.array(real_label)
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()

    predict_label = np.array(predict_label)
    real_label = np.array(real_label)
    result = compute_metrics("csc", predict_label, real_label)
    # results.update(result)
    logger.info("\n")
    logger.info("***** Eval results {} *****".format(data_type))
    logger.info("precision = {}, recall = {}, f1 = {}".format(
        result["precision"], result["recall"], result["f1"]))

    results.update(result)

    return results


# 加载数据
def load_features(file_path, is_file_path=True, is_test=False):
    if is_file_path:
        logger.info("start load file: {}".format(file_path))
        features = torch.load(file_path)
    else:
        logger.info("start load features")
        features = file_path

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)

    if not is_test:
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_noise_ids = torch.tensor([f.noise_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_lens, all_label_ids, all_noise_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_lens)

    if is_file_path:
        logger.info("finish load file: {}".format(file_path))

    else:
        logger.info("finish load features")

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='model dir path')
    parser.add_argument("--env_pos", default="0", type=str,
                        help="cuda pos")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="cuda pos")
    parser.add_argument("--calc_rank", default=0, type=int,
                        help="evaluate cuda pos")
    # parser.add_argument("--data_dir", default=None, type=str,
    # 					help="data type")
    # parser.add_argument("--data_name", default=None, type=str,
    # 					help="data name")
    parser.add_argument("--out_dir", default="./fine_tune_model", type=str,
                        help="save model dir")
    parser.add_argument("--no_add", default=3, type=int,
                        help="no add")
    parser.add_argument("--num_train_epochs", default=15, type=int,
                        help="num_train_epochs")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    # parser.add_argument("--vocab_file", type = str, default = "vocab.txt",
    # 					help = "vocab file path")
    parser.add_argument("--warmup_proportion", type=float, default=0.02,
                        help="warm ratio")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight_decay")
    parser.add_argument("--use_cuda", action="store_true",
                        help="use cuda or not")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--alpha", default=0.3, type=float,
                        help="kl_loss' ratio")
    parser.add_argument("--retrain", action="store_true",
                        help="re train")
    parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
    parser.add_argument("--valid_data_path", type=str, default="./data/valid.txt")
    parser.add_argument("--cached_file_dir", type=str, default="./data/cached")
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.env_pos)
        if torch.cuda.is_available() and args.use_cuda:
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
            args.use_cuda = False

    args.device = device

    tokenizer = FullTokenizer(vocab_file=os.path.join(args.model_dir,
                                                      "vocab.txt"), do_lower_case=False)
    config = FuncunConfig.from_pretrained(os.path.join(args.model_dir,
                                                       "bert_config.json"))

    config.alpha = args.alpha

    init_logger(os.path.join(args.out_dir, "train.log"),
                log_file_level=logging.INFO)
    seed_everything(args.seed)

    logger.info("Training config parameters {}".format(config))
    logger.info("Training args parameters {}".format(args))

    model = FuncunCRASpell.from_pretrained(args.model_dir,
                                           config=config)

    model.to(args.device)

    num_gpus = torch.cuda.device_count()
    if args.local_rank != -1 and num_gpus > 1:
        # model = 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    train(args, model, tokenizer)
