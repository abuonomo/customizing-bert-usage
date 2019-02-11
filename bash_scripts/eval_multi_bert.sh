#!/bin/bash
export BERT_BASE_DIR=$(pwd)/cased_L-12_H-768_A-12
export GLUE_DIR=$(pwd)/glue_data
export TRAINED_CLASSIFIER=$(pwd)/experiments/multi_test/model.ckpt-11.index

python bert/run_classifier.py \
  --task_name=STI \
  --do_predict=true \
  --data_dir=$(pwd)/experiments/multi_test/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=$(pwd)/experiments/multi_test/
