#!/bin/bash

data="flex_attention_data_a100.txt"

python train_decision_flex_attention.py ${data} --heuristic-name FlexAttentionA100 --gpu A100
