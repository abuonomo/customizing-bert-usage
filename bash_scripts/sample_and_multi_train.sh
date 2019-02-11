#!/bin/bash
export BERT_BASE_DIR=$(pwd)/cased_L-12_H-768_A-12
export OUTDIR="experiments/multi_test"
export MIN_TERM_THRESHOLD=500
export SHRINK_FACTOR=0.001

python format_multi_ml_sets.py results_merged.json ${OUTDIR} ${SHRINK_FACTOR} ${MIN_TERM_THRESHOLD}

python bert/run_classifier.py \
  --task_name=STI \
  --do_train=true \
  --do_eval=true \
  --data_dir="$(pwd)/${OUTDIR}" \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir="$(pwd)/${OUTDIR}"

python get_checkpoint.py "${OUTDIR}" "${OUTDIR}/max_checkpoint.txt"
read -r checkpoint<"${OUTDIR}/max_checkpoint.txt"
export TRAINED_CLASSIFIER="${OUTDIR}/${checkpoint}"

python bert/run_classifier.py \
  --task_name=STI \
  --do_predict=true \
  --data_dir="${OUTDIR}" \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir="${OUTDIR}"
