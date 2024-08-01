#!/bin/bash

data="flexattention_data_h100.txt"

python train_decision_flex_attention.py ${data} --heuristic-name FlexAttentionH100 --gpu H100
