export BS=16

XRT_TPU_CONFIG="tpu_worker;0;$COLAB_TPU_ADDR" \
python xla_spawn.py --num_cores=8 run_mlm.py \
--seed 42 \
--model_type exbert \
--tokenizer_name exbert \
--train_file paws_corpus.txt \
--num_train_epochs 10 \
--per_device_train_batch_size $BS \
--per_device_eval_batch_size $BS \
--do_train \
--output_dir ./exbert-mlm \
--logging_first_step \
--max_seq_length 300 \
