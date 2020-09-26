#!/bin/bash

OUTFILE=sparse-test.txt
PYTORCH_HOME=$HOME/gitrepos/pytorch

echo "!! SPARSE SPMV TIME BENCHMARK!! " >> $OUTFILE
echo "" >> $OUTFILE

echo "----- USE_MKL=1 -----" >> $OUTFILE
cd $PYTORCH_HOME
rm -rf build

export USE_MKL=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build

python setup.py install
cd benchmarks
for dim0 in 1000 5000 10000; do
    for nnzr in 0.01 0.05 0.1 0.3; do
        python -m sparse.spmv --format gcs --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
        python -m sparse.spmv --format coo --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
    done
done
echo "----------------------" >> $OUTFILE

echo "----- USE_MKL=0 ------" >> $OUTFILE
cd $PYTORCH_HOME
rm -rf build

export USE_MKL=0
python setup.py install

cd benchmarks
for dim0 in 1000 5000 10000; do
    for nnzr in 0.01 0.05 0.1 0.3; do
        python -m sparse.spmv --format gcs --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
        python -m sparse.spmv --format coo --m $dim0 --nnz_ratio $nnzr --outfile $OUTFILE
    done
done
echo "----------------------" >> $OUTFILE

