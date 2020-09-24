#!/bin/bash

OUTFILE=sparse-test.txt
PYTORCH_HOME=$HOME/gitrepos/pytorch

echo "----- USE_MKL=1 -----" >> $OUTFILE
cd $PYTORCH_HOME
export USE_MKL=1
python setup.py install
cd benchmarks
for 1000 5000 10000 in dim0; do
    for 0.01 0.05 0.1 0.3 in nnzr; do
        python -m sparse.spmv --format gcs --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
        python -m sparse.spmv --format coo --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
    done
done



