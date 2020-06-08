#!/usr/bin/env bash

python run_classifier.py \
--task_name pc \
--do_train \
--do_eval \
--eval_while_training \
--data_dir data/predicate-classification \
--bert_model bert-base-multilingual-uncased \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--percent 100 \
--output_dir results/50binary_LSTM8_1FC \
--use_predicate_indicator