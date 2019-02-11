#!/bin/bash
export BERT_BASE_DIR=$(pwd)/cased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/tmp/mrpc_output2/model.ckpt-166.index

python bert/run_classifier.py \
  --task_name=STI \
  --do_predict=true \
  --data_dir=STI_small/test \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output2/

