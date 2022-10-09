# CCL2022-track1

## 数据增强

cd data_augmentation
sh run_aug.sh
其中 data_augementation/data/train_aug.txt为增强后的数据.
如果需要原始的文本数据请从这里 https://pan.baidu.com/s/1fHS75lH7GU_hqG-mA1YKqg 下载。提取码: as6j

## 文本纠错

#### CRA-Spell

```bash
训练 
python train.py \
--model_dir ./chinese_roberta_wwm_ext_pytorch/ \
--local_rank -1 \
--per_gpu_train_batch_size 32 \
--num_train_epochs 20 \
--use_cuda 

预测 
python predict.py \
--model_dir ./finetune_model/checkpoint-20/ \
--env_pos 0 \
--local_rank -1 \
--use_cuda

其中训练时用到的候选集可以从这里下载[]()
```

#### macbert4csc

```bash
cd macbert4csc
训练 python train.py
预测 python infer.py --text_file ../yaclc-csc_test.src --save_path ../predict/roberta_01.txt
```

#### ReaLiSe

```bash
cd ReaLiSe
训练 sh train.sh
预测 sh test.sh
```

#### RoBERTa_01

```bash
cd RoBERTa_01
训练 sh pipe.sh
预测 sh predict.sh
```

## 模型结果合并

合并各个模型的检错结果，然后利用候选集进行纠错。

```
python ensemble.py
```
