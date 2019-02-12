#!/usr/bin/env bash
python ../../src/format_ml_sets.py ../../data/datasets/results_merged.json ../data/ 2 0.01 --max_threshold=500
