#!/usr/bin/env bash

python run_classifier.py \
--task_name pc \
--do_predict \
--data_dir data/predicate-classification \
--bert_model bert-base-multilingual-uncased \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 16 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--num_train_epochs 50 \
--percent 100 \
--output_dir Classifier_model/50binary_BiLSTM8_1FC/ \
--detector_prediction_file Detector_model/extra-train-ed-mbert-cased-combined-5ep/detector_test_predictions.txt \
--use_predicate_indicator
