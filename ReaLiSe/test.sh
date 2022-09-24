DATE_DIR="data"
CKPT_DIR="output_old/saved_ckpt-5000"
OUTPUT_DIR="output_predict"

python src/test.py \
    --device "cuda:0" \
    --ckpt_dir $CKPT_DIR \
    --data_dir $DATE_DIR \
    --testset_year 22 \
    --ckpt_num -1 \
    --output_dir $OUTPUT_DIR
