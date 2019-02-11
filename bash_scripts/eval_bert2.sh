#!/bin/bash
export BERT_BASE_DIR=$(pwd)/cased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=space_sciences/tf/model.ckpt-12504.index

python bert/run_classifier.py \
  --task_name=STI \
  --do_predict=true \
  --data_dir=experiments/test_results/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=space_sciences/tf/

