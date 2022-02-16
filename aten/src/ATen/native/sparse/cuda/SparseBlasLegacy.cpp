/*
Functions here use deprecated cuSPARSE API that was removed in CUDA 11.
This file will be removed eventually.
*/
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>

namespace at {
namespace native {

void s_addmm_out_csr_sparse_dense_cuda_worker(int64_t nnz, int64_t m, int64_t n, int64_t k, const Tensor& r_, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, const Tensor& dense) {
  TORCH_INTERNAL_ASSERT(nnz > 0);

  // No half support, so we don't have to use CUDATypeConversion
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(), "addmm_sparse_cuda", [&] {
        scalar_t cast_beta = beta.to<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();
        Tensor r__;
        if (cast_beta == scalar_t(0)) {
          r_.zero_();
        } else if (!at::sparse::is_same_tensor(t, r_)) {
          r_.copy_(t);
        }
        if (r_.stride(0) == 1 && r_.stride(1) == r_.size(0)) {
          r__ = r_;
        } else {
          // Note: This storage arrangement is preferred due to most of the CUDA kernels handle only contiguous tensors
          r__ = r_.transpose(0, 1).clone(at::MemoryFormat::Contiguous);
          r__.transpose_(0, 1);
        }
        TORCH_INTERNAL_ASSERT(r__.mT().is_contiguous());
        Tensor dense_;
        char transpose_dense;
        if (dense.stride(0) == 1 && dense.stride(1) == dense.size(0)) {
          transpose_dense = 'n';
          dense_ = dense;
        } else if (dense.stride(1) == 1 && dense.stride(0) == dense.size(1)) {
          transpose_dense = 't';
          dense_ = dense;
        } else {
          transpose_dense = 't';
          dense_ = dense.contiguous();
        }

        sparse::cuda::csrmm2(
          'n',
          transpose_dense,
          m,
          n,
          k,
          nnz,
          cast_alpha,
          values.data_ptr<scalar_t>(),
          crow_indices.data_ptr<int32_t>(),
          col_indices.data_ptr<int32_t>(),
          dense_.data_ptr<scalar_t>(),
          (transpose_dense == 'n' ? dense_.stride(1) : dense_.stride(0)),
          cast_beta,
          r__.data_ptr<scalar_t>(),
          r__.stride(1));

        if (!at::sparse::is_same_tensor(r__, r_)) {
          r_.copy_(r__);
        }
      }
    );
}

} // namespace native
} // namespace at
