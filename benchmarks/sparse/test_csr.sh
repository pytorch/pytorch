OUTFILE=spmm-no-mkl-test.txt
PYTORCH_HOME=$1

cd $PYTORCH_HOME

echo "" >> $OUTFILE
echo "----- USE_MKL=1 -----" >> $OUTFILE
rm -rf build

export USE_MKL=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build

python setup.py install

cd benchmarks
echo "!! SPARSE SPMM TIME BENCHMARK!! " >> $OUTFILE
for dim0 in 1000 5000 10000; do
    for nnzr in 0.01 0.05 0.1 0.3; do
        python -m sparse.spmm --format csr --m $dim0 --n $dim0 --k $dim0 --nnz-ratio $nnzr --outfile $OUTFILE
        # python -m sparse.spmm --format coo --m $dim0 --n $dim0 --k $dim0 --nnz-ratio $nnzr --outfile $OUTFILE
    done
done
echo "----------------------" >> $OUTFILE

cd $PYTORCH_HOME
echo "----- USE_MKL=0 ------" >> $OUTFILE
rm -rf build

export USE_MKL=0
python setup.py install

cd benchmarks
for dim0 in 1000 5000 10000; do
    for nnzr in 0.01 0.05 0.1 0.3; do
        python -m sparse.spmv --format csr --m $dim0 --nnz-ratio $nnzr --outfile $OUTFILE
        python -m sparse.spmv --format coo --m $dim0 --nnz-ratio $nnzr --outfile $OUTFILE
    done
done
echo "----------------------" >> $OUTFILE
