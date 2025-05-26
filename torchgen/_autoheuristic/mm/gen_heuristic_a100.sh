#!/bin/bash

dir="a100/"
data="a100_mm.txt"
python train_decision_mm.py ${dir}a100_mm.txt --heuristic-name MMRankingA100 --ranking 10 --save-dot --data train_timm ${dir}a100_timm_train_mm.txt --data train_hf ${dir}a100_hf_train_mm.txt
