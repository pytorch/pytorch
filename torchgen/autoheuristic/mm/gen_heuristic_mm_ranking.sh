#!/bin/bash

data="a100_mm.txt"

python train_decision_mm.py a100_mm.txt  --heuristic-name MMRankingA100 --data mm_hf mm_hf.txt --data mm_timm mm_timm.txt --data mm_torchbench mm_torchbench.txt --ranking --save-dot
