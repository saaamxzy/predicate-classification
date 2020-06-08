#!/usr/bin/env bash

python run_classifier.py \
--task_name pc \
--do_eval \
--data_dir data/predicate-classification \
--bert_model bert-base-multilingual-cased \
--max_seq_length 128 \
--train_batch_size 16 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--percent 100 \
--output_dir results/no_binary_2LSTM_4096FC_no_pooling
