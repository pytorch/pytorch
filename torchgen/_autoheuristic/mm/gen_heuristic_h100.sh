#!/bin/bash

data="h100_mm.txt"
dir="h100/"

python train_decision_mm.py ${dir}${data} --heuristic-name MMRankingH100 --ranking --data train_timm ${dir}timm_train_mm.txt --data train_hf ${dir}hf_train_mm.txt
