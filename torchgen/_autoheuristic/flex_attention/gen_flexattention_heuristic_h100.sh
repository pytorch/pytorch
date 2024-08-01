#!/bin/bash

data="h100-flex-data.txt"

python train_decision_flex_attention.py ${data} --heuristic-name FlexAttentionH100 --gpu H100
