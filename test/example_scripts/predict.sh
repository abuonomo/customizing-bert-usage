#!/usr/bin/env bash
python ../../src/helper_classes.py ../data/engine_tests/ \
    ../../cased_L-12_H-768_A-12/bert_config.json \
    ../../cased_L-12_H-768_A-12/vocab.txt \
    ../data/engine_tests/tf --do_predict
