#!/usr/bin/env bash

python run_classifier.py \
--task_name pc \
--do_train \
--do_eval \
--do_lower_case \
--data_dir data/first_quad \
--bert_model bert-base-uncased \
--max_seq_length 128 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--percent 100 \
--output_dir output
