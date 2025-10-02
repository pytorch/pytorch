#!/bin/bash

dir="h100/"
data="h100_mm.txt"
python train_decision_mm.py ${dir}h100_mm.txt --heuristic-name MMRankingH100 --ranking 10 --save-dot --data train_timm ${dir}h100_timm_train_mm.txt --data train_hf ${dir}h100_hf_train_mm.txt
