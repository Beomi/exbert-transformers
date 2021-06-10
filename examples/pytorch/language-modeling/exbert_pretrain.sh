export BS=2
export NCCL_DEBUG=INFO

python run_mlm.py \
--seed 42 \
--model_type exbert \
--tokenizer_name exbert \
--train_file ~/Downloads/85.txt \
--num_train_epochs 1 \
--per_device_train_batch_size $BS \
--per_device_eval_batch_size $BS \
--do_train \
--output_dir ./exbert-mlm \
--logging_first_step \
--max_seq_length 300 \
