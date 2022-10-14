PRETRAINED_DIR="pretrained"
DATE_DIR="data"
OUTPUT_DIR="output"

python -m torch.distributed.launch --master_port=453$(($RANDOM%90+10)) --nproc_per_node=2 src/run.py \
    --model_type bert-pho2-res-arch3 \
    --model_name_or_path $PRETRAINED_DIR \
    --image_model_type 0 \
    --output_dir $OUTPUT_DIR  \
    --do_train --do_eval --do_predict  \
    --data_dir $DATE_DIR \
    --train_file train/src.txt \
    --dev_file train/train.pkl \
    --dev_label_file dev/lbl.txt \
    --predict_file dev/dev.pkl \
    --predict_label_file dev/yaclc-csc_dev.lbl \
    --order_metric sent-detect-f1  \
    --metric_reverse  \
    --num_save_ckpts 5 \
    --remove_unused_ckpts  \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8  \
    --learning_rate 5e-5 \
    --num_train_epochs 1000  \
    --seed 17 \
    --warmup_steps 10000  \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --resfonts font3_fanti