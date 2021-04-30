#pragma once

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>

namespace at { namespace native {

void s_addmm_out_csr_sparse_dense_cuda_worker(int64_t nnz, int64_t m, int64_t n, int64_t k, Tensor& r_, const Scalar& beta, const Tensor& t, const Scalar& alpha, Tensor& crow_indices, Tensor& col_indices, Tensor& values, const Tensor& dense);

void s_addmm_out_sparse_dense_cuda_worker(int64_t nnz, int64_t m, int64_t n, int64_t k, Tensor& r_, const Scalar& beta, const Tensor& t, const Scalar& alpha, Tensor& indices, Tensor& values, const Tensor& dense);

}} // namespace at::native
