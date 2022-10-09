python ./decode.py \
--pretrained_model ./chinese_roberta_wwm_large_ext_L-24_H-1024_A-16_torch \
--test_path ./yaclc-csc_test.src \
--model_path ./exps/sighan/sighan-epoch-11.pt \
--save_path ../predict/roberta_01.txt