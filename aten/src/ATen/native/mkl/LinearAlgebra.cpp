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
  const float** A, const int lda, const float** B, const int ldb, const float beta,
  float** C, const int ldc) {

  cblas_sgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const double alpha,
  const double** A, const int lda, const double** B, const int ldb, const double beta,
  double** C, const int ldc) {

  cblas_dgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

template <typename scalar_t>
static inline void baddbmm_mkl_template(const Tensor& res, const Tensor& mat1, const Tensor& mat2, Scalar beta_, Scalar alpha_) {
  auto is_transposed = [&](const TensorAccessor<scalar_t, 2>& t) {
    return t.stride(0) == 1 && t.stride(1) >= t.size(0);
  };

  auto mat1_acc = mat1.accessor<scalar_t, 3>();
  auto mat2_acc = mat2.accessor<scalar_t, 3>();
  auto res_acc = res.accessor<scalar_t, 3>();

  const CBLAS_TRANSPOSE trans_A = is_transposed(mat1_acc[0]) ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_B = is_transposed(mat2_acc[0]) ? CblasTrans : CblasNoTrans;

  const int batch_size = mat1_acc.size(0);
  const int M = mat1_acc.size(1);
  const int N = mat2_acc.size(2);
  const int K = mat1_acc.size(2);
  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  const int lda = is_transposed(mat1_acc[0]) ? mat1_acc[0].stride(1) : mat1_acc[0].stride(0);
  const int ldb = is_transposed(mat2_acc[0]) ? mat2_acc[0].stride(1) : mat2_acc[0].stride(0);
  const int ldc = res[0].stride(0);

  std::vector<const scalar_t*> A(batch_size);
  std::vector<const scalar_t*> B(batch_size);
  std::vector<scalar_t*> C(batch_size);

  for (int64_t batch = 0; batch < batch_size; batch++) {
    A[batch] = mat1_acc[batch].data();
    B[batch] = mat2_acc[batch].data();
    C[batch] = res_acc[batch].data();
  }

  gemm_batched(trans_A, trans_B, batch_size, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
}

Tensor& _baddbmm_mkl_(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  // checks are done in native/LinearAlgebra.cpp
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "baddbmm__mkl", [&] {
      baddbmm_mkl_template<scalar_t>(self, batch1, batch2, beta, alpha);
    });

  return self;
}

}} // namespace at::native

#endif
