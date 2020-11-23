#!/bin/bash

OUTFILE=spmm-no-mkl-test.txt
DATASET_ROOT_DIR=$HOME/datasets/

# wget https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz -P $DATASET_ROOT_DIR
# tar -xvf $DATASET_ROOT_DIR/dlmc.tar.gz 

echo "!! SPARSE SPMS TIME BENCHMARK!! " 

python dlmc_bench.py -p $DATASET_ROOT_DIR/dlmc/rn50 -d random_pruning -o /tmp/matmul_bench.pkl
python plot_results.py -i /tmp/matmul_bench.pkl