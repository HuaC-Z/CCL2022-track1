# CCL2022-track1

## 数据连接

data_augmentation 该目录为数据增强

本项目使用的训练数据的链接: https://pan.baidu.com/s/1LQ2MI9G789RWUTVnk3CetA 提取码: dsbg

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
预训练模型链接：https://pan.baidu.com/s/1XKUtbcwax9C05oovkERPBg 提取码：6666 
```

#### macbert4csc

```bash
cd macbert4csc
训练 python train.py
预测 python infer.py --text_file ../yaclc-csc_test.src --save_path ../predict/roberta_01.txt
预训练模型链接: https://pan.baidu.com/s/1bit6Htu2QKUzzR8X1D9sJA 提取码: btgw
```

#### ReaLiSe

```bash
cd ReaLiSe
python change_data_format.py 调整数据格式
训练 sh train.sh
预测 sh test.sh
预训练模型链接：https://pan.baidu.com/s/1N8kro7dAGgmumi9YS_nN7A?pwd=6666 提取码：6666
```

#### RoBERTa_01

```bash
cd RoBERTa_01
python change_data_format.py 调整数据格式
训练 sh pipe.sh
预测 sh predict.sh
预训练模型链接: https://pan.baidu.com/s/1zX8s1RB3CTlRXGRcWmwKZg 提取码: 1v2p
```

## 模型结果合并

合并各个模型的检错结果，然后利用候选集进行纠错。

```
python ensemble.py
```

各个模型使用的都是论文原始环境。
