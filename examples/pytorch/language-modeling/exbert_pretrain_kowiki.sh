export BS=16
export NCCL_DEBUG=INFO

python run_mlm.py \
--seed 42 \
--model_type exbert \
--tokenizer_name exbert \
--train_file kowikitext.txt \
--num_train_epochs 2 \
--per_device_train_batch_size $BS \
--per_device_eval_batch_size $BS \
--do_train \
--output_dir ./exbert-kowiki \
--logging_first_step \
--max_seq_length 300 \
