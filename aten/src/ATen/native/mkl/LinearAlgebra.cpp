#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {

Tensor& _baddbmm_mkl_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
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

static inline void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int  M, const int N, const int K, const float alpha, const float* A,
  const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc) {
  cblas_sgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int  M, const int N, const int K, const double alpha, const double* A,
  const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc) {
  cblas_dgemm(CblasRowMajor, trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int  M, const int N, const int K, const c10::complex<float> alpha,
  const c10::complex<float>* A, const int lda, const c10::complex<float>* B, const int ldb,
  const c10::complex<float> beta, c10::complex<float>* C, const int ldc) {
  cblas_cgemm(CblasRowMajor, trans_A, trans_B, M, N, K, reinterpret_cast<const void *>(&alpha),
    reinterpret_cast<const void*>(A), lda, reinterpret_cast<const void*>(B), ldb,
    reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(C), ldc);
}

static inline void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int  M, const int N, const int K, const c10::complex<double> alpha,
  const c10::complex<double>* A, const int lda, const c10::complex<double>* B, const int ldb,
  const c10::complex<double> beta, c10::complex<double>* C, const int ldc) {
  cblas_zgemm(CblasRowMajor, trans_A, trans_B, M, N, K, reinterpret_cast<const void *>(&alpha),
    reinterpret_cast<const void*>(A), lda, reinterpret_cast<const void*>(B), ldb,
    reinterpret_cast<const void*>(&beta), reinterpret_cast<void*>(C), ldc);
}

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

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const c10::complex<float> alpha,
  const c10::complex<float>** A, const int lda, const c10::complex<float>** B, const int ldb,
  const c10::complex<float> beta, c10::complex<float>** C, const int ldc) {

  cblas_cgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, reinterpret_cast<const void*>(&alpha),
    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const c10::complex<double> alpha,
  const c10::complex<double>** A, const int lda, const c10::complex<double>** B, const int ldb,
  const c10::complex<double> beta, c10::complex<double>** C, const int ldc) {

  cblas_zgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, reinterpret_cast<const void*>(&alpha),
    reinterpret_cast<const void**>(A), &lda, reinterpret_cast<const void**>(B), &ldb,
    reinterpret_cast<const void*>(&beta), reinterpret_cast<void**>(C), &ldc, 1, &batch_size);
}

template <typename scalar_t>
static inline void baddbmm_mkl_template(const Tensor& res, const Tensor& mat1, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  auto is_transposed = [](const c10::IntArrayRef& strides, const c10::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };

  const CBLAS_TRANSPOSE trans_A =
      is_transposed(mat1_strides, mat1_sizes) ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_B =
      is_transposed(mat2_strides, mat2_sizes) ? CblasTrans : CblasNoTrans;


  // mat1: batch_size * M * K
  const int batch_size = mat1_sizes[0];
  const int M = mat1_sizes[1];
  // mat2: batch_size * K * N
  const int N = mat2_sizes[2];
  const int K = mat1_sizes[2];

  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  const int lda = trans_A == CblasTrans ? mat1_strides[2] : mat1_strides[1];
  const int ldb = trans_B == CblasTrans ? mat2_strides[2] : mat2_strides[1];
  const int ldc = res.strides()[1];

  // avoid using tensor accessor in the case of mat1/mat2 not being transposed
  // or only transposed in the last two axes
  const bool canAvoidTensorAccessor = mat1_strides[0] == mat1_sizes[1] * mat1_sizes[2] &&
    mat2_strides[0] == mat2_sizes[1] * mat2_sizes[2];

  scalar_t* const res_data = res.data_ptr<scalar_t>();

  if (batch_size == 1) {
    const scalar_t* A;
    const scalar_t* B;
    if (canAvoidTensorAccessor) {
      scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
      scalar_t* mat2_data = mat2.data_ptr<scalar_t>();
      A = mat1_data;
      B = mat2_data;
    } else {
      auto mat1_acc = mat1.accessor<scalar_t, 3>();
      auto mat2_acc = mat2.accessor<scalar_t, 3>();
      A = mat1_acc[0].data();
      B = mat2_acc[0].data();
    }
    gemm(trans_A, trans_B, M, N, K, alpha, A, lda, B, ldb, beta, res_data, ldc);
    return;
  }

  std::vector<const scalar_t*> A;
  A.reserve(batch_size);
  std::vector<const scalar_t*> B;
  B.reserve(batch_size);
  std::vector<scalar_t*> C;
  C.reserve(batch_size);

  // avoid using tensor accessor in the case of mat1/mat2 not being transposed
  // or only transposed in the last two axis
  const auto res_sizes = res.sizes();
  if (canAvoidTensorAccessor) {
    scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
    scalar_t* mat2_data = mat2.data_ptr<scalar_t>();
    for (int64_t batch = 0; batch < batch_size; batch++) {
      A.emplace_back(mat1_data + batch * mat1_sizes[1] * mat1_sizes[2]);
      B.emplace_back(mat2_data + batch * mat2_sizes[1] * mat2_sizes[2]);
      C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
    }
  } else {
    auto mat1_acc = mat1.accessor<scalar_t, 3>();
    auto mat2_acc = mat2.accessor<scalar_t, 3>();
    for (int64_t batch = 0; batch < batch_size; batch++) {
      A.emplace_back(mat1_acc[batch].data());
      B.emplace_back(mat2_acc[batch].data());
      C.emplace_back(res_data + batch * res_sizes[1] * res_sizes[2]);
    }
  }

  gemm_batched(trans_A, trans_B, batch_size, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
}

Tensor& _baddbmm_mkl_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // checks are done in native/LinearAlgebra.cpp
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "baddbmm__mkl", [&] {
      baddbmm_mkl_template<scalar_t>(self, batch1, batch2, beta, alpha);
    });

  return self;
}

}} // namespace at::native

#endif
