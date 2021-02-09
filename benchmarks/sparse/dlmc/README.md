# Sparse benchmarks

These sets of benchmarks are for the sparse matrix functionality using a popular real dataset collection called 
the Deep Learning Matrix Collection (DLMC), which were used in recent studies [1, 2].

Performance benchmarks scripts for matrix-matrix and matrix-vector ops 
(dense-sparse, sparse-sparse, and compare to dense-dense) are implemented here.

- `spmm.py` this benchmarks is for  Sparse matrix-matrix multiplication (SPMM) performance test, for both `sparse @ sparse` and `sparse @ dense` operations. Both can run in forward and backward mode, on CPU or CUDA, using different datasets from the dataset collection DLMC.  

- `spmv.py`  this benchmark is for Sparse matrix-vector multiplication (SPMV) performance test.  

References:

1. Trevor Gale, Matei Zaharia, Cliff Young, Erich Elsen. Sparse GPU Kernels for Deep Learning. 
Proceedings of the International Conference for High Performance Computing, 2020. 
https://github.com/google-research/google-research/tree/master/sgk

2. Trevor Gale, Erich Elsen, Sara Hooker. The State of Sparsity in Deep Neural Networks. 
https://github.com/google-research/google-research/tree/master/state_of_sparsity
