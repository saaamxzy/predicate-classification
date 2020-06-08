#!/usr/bin/env bash

python evaluate_results.py \
--true_data_file data/event-detection-classification/test.txt \
--pred_data_file Classifier_model/50binary_BiLSTM8_1FC/final_predictions.txt
