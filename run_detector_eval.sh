echo "evaluating event trigger detector"
export BERT_MODEL=bert-base-multilingual-cased
export MAX_LENGTH=512
export BATCH_SIZE=4
export NUM_EPOCHS=5
export SAVE_STEPS=3000
export SEED=1
export OUTPUT_DIR=Detector_model/extra-train-ed-mbert-cased-combined-5ep
export TEST_FILE=test
python run_ner.py \
--data_dir data/event-detection \
--model_type bert \
--labels data/event-detection/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_predict \
--test_file $TEST_FILE \
