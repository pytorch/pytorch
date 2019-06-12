#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {

Tensor& _baddbmm_mkl_(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  AT_ERROR("bmm: ATen not compiled with MKL support");
}

}}

#else // AT_MKL_ENABLED

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <mkl.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>

namespace at { namespace native {

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const float alpha,
  const float** A, const float** B, const float beta, float** C) {
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;

  cblas_sgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const double alpha,
  const double** A, const double** B, const double beta, double** C) {
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;

  cblas_dgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

template <typename scalar_t>
static inline void baddbmm_mkl_template(const Tensor& res, const Tensor& mat1, const Tensor& mat2, Scalar beta_, Scalar alpha_) {
  auto is_transposed = [&](const Tensor& t) {
    return t.stride(0) == 1 && t.stride(1) == t.size(0);
  };
  const CBLAS_TRANSPOSE trans_A = is_transposed(mat1[0]) ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_B = is_transposed(mat2[0]) ? CblasTrans : CblasNoTrans;

  const int batch_size = mat1.size(0);
  const int M = mat1.size(1);
  const int N = mat2.size(2);
  const int K = mat1.size(2);
  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  std::vector<const scalar_t*> A(batch_size);
  std::vector<const scalar_t*> B(batch_size);
  std::vector<scalar_t*> C(batch_size);
  for (int64_t batch = 0; batch < batch_size; batch++) {
    A[batch] = mat1[batch].data<scalar_t>();
    B[batch] = mat2[batch].data<scalar_t>();
    C[batch] = res[batch].data<scalar_t>();
  }

  gemm_batched(trans_A, trans_B, batch_size, M, N, K, alpha, A.data(), B.data(), beta, C.data());
}

Tensor& _baddbmm_mkl_(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  // checks are done in native/LinearAlgebra.cpp
  AT_DISPATCH_FLOATING_TYPES(self.type(), "baddbmm__mkl", [&] {
      baddbmm_mkl_template<scalar_t>(self, batch1, batch2, beta, alpha);
    });

  return self;
}

}} // namespace at::native

#endif
