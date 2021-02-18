#!/bin/bash

DATASET_ROOT_DIR=$HOME/datasets/

# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR
# tar -xvf $DATASET_ROOT_DIR/dlmc.tar.gz 

echo "!! SPARSE SPMS TIME BENCHMARK!! " 

python -m dlmc.matmul_bench  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation sparse@sparse
python -m dlmc.matmul_bench  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation sparse@sparse --backward_test

python -m dlmc.matmul_bench  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation sparse@dense
python -m dlmc.matmul_bench  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation sparse@dense --backward_test

python -m dlmc.matmul_bench  --path $DATASET_ROOT_DIR/dlmc/rn50 --dataset random_pruning --operation sparse@vector 
