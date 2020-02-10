// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different backends. Notably:
// (1) For all BLAS-related functions, one can explicitly request a BLAS backend
//     such as MKL, openblas or Atlas. To see the set of supported backends
//     currently provided, check //third_party/blas/.
// (2) If one chooses to link against MKL, we utilize MKL's vector math library
//     (VML) for a few functions such as Exp and Log.
// (3) Fallback implementations are provided in Eigen for cross-platform
//     support. Since Eigen is a header-only library and supports a number of
//     platforms, it allows one to quickly port Caffe2 to different platforms
//     where BLAS may not be present.

#include "caffe2/utils/math.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/fixed_divisor.h"

#include "Eigen/Core"
#include "Eigen/Dense"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#ifdef CAFFE2_USE_HPTT
#include <hptt.h>
#endif // CAFFE2_USE_HPTT

#if defined(_MSC_VER)
#include <process.h>
#endif

namespace caffe2 {

namespace math {

////////////////////////////////////////////////////////////////////////////////
// BLAS alternatives.
// Depending on whether we have specified an external BLAS library or not, we
// will delegate the Caffe math functions that are BLAS-related to either the
// CBLAS call or the Eigen implementation.
////////////////////////////////////////////////////////////////////////////////
#ifdef CAFFE2_USE_EIGEN_FOR_BLAS

// Caffe2 gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
//
// The gemm call implements the following operation:
//
//                  C = alpha * op(A) * op(B) + beta * C
//
// where op(A) has size M x K, op(B) has size K x N, and C has size M x N. Each
// of A, B, and C are matrices and alpha and beta are scalars. Note that the
// most common use case of gemm will involve setting alpha to 1 and beta to 0.
//
// op(A) and op(B) represent the transformations that are done to A and B before
// the matrix multiply; depending on the flags set, op(A) is equal to A or A^T
// (transpose) if the argument TransA or TransB is set to CblasNoTrans or
// CblasTrans, respectively, for each of A and B.
template <>
C10_EXPORT void Gemm<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CPUContext* context,
    TensorProto::DataType math_type) {
  auto C_mat = EigenMatrixMap<float>(C, N, M);
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (trans_A) {
    case CblasNoTrans: {
      switch (trans_B) {
        case CblasNoTrans:
          C_mat.noalias() += alpha *
              (ConstEigenMatrixMap<float>(B, N, K) *
               ConstEigenMatrixMap<float>(A, K, M));
          return;
        case CblasTrans:
          C_mat.noalias() += alpha *
              (ConstEigenMatrixMap<float>(B, K, N).transpose() *
               ConstEigenMatrixMap<float>(A, K, M));
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    case CblasTrans: {
      switch (trans_B) {
        case CblasNoTrans:
          C_mat.noalias() += alpha *
              (ConstEigenMatrixMap<float>(B, N, K) *
               ConstEigenMatrixMap<float>(A, M, K).transpose());
          return;
        case CblasTrans:
          C_mat.noalias() += alpha *
              (ConstEigenMatrixMap<float>(B, K, N).transpose() *
               ConstEigenMatrixMap<float>(A, M, K).transpose());
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_A";
  }
}

template <>
C10_EXPORT void GemmEx<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int lda,
    const float* B,
    const int ldb,
    const float beta,
    float* C,
    const int ldc,
    CPUContext*) {
  EigenOuterStridedMatrixMap<float> C_mat(C, N, M, EigenOuterStride(ldc));
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (trans_A) {
    case CblasNoTrans: {
      switch (trans_B) {
        case CblasNoTrans:
          C_mat.noalias() += alpha *
              (ConstEigenOuterStridedMatrixMap<float>(
                   B, N, K, EigenOuterStride(ldb)) *
               ConstEigenOuterStridedMatrixMap<float>(
                   A, K, M, EigenOuterStride(lda)));
          return;
        case CblasTrans:
          C_mat.noalias() += alpha *
              (ConstEigenOuterStridedMatrixMap<float>(
                   B, K, N, EigenOuterStride(ldb))
                   .transpose() *
               ConstEigenOuterStridedMatrixMap<float>(
                   A, K, M, EigenOuterStride(lda)));
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    case CblasTrans: {
      switch (trans_B) {
        case CblasNoTrans:
          C_mat.noalias() += alpha *
              (ConstEigenOuterStridedMatrixMap<float>(
                   B, N, K, EigenOuterStride(ldb)) *
               ConstEigenOuterStridedMatrixMap<float>(
                   A, M, K, EigenOuterStride(lda))
                   .transpose());
          return;
        case CblasTrans:
          C_mat.noalias() += alpha *
              (ConstEigenOuterStridedMatrixMap<float>(
                   B, K, N, EigenOuterStride(ldb))
                   .transpose() *
               ConstEigenOuterStridedMatrixMap<float>(
                   A, M, K, EigenOuterStride(lda))
                   .transpose());
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_B";
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for trans_A";
  }
}

template <>
C10_EXPORT void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* context,
    TensorProto::DataType math_type) {
  EigenVectorMap<float> y_vec(y, trans_A == CblasNoTrans ? M : N);
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float values. As a result, if beta is 0, we explicitly do a setzero.
    y_vec.setZero();
  } else {
    y_vec *= beta;
  }
  switch (trans_A) {
    case CblasNoTrans: {
      y_vec.noalias() += alpha *
          (ConstEigenMatrixMap<float>(A, N, M).transpose() *
           ConstEigenVectorMap<float>(x, N));
      return;
    }
    case CblasTrans: {
      y_vec.noalias() += alpha *
          (ConstEigenMatrixMap<float>(A, N, M) *
           ConstEigenVectorMap<float>(x, M));
      return;
    }
    default:
      LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input.";
  }
}

#define CAFFE2_SPECIALIZED_DOT(T)                                        \
  template <>                                                            \
  C10_EXPORT void Dot<T, CPUContext>(                                    \
      const int N, const T* a, const T* b, T* y, CPUContext* context) {  \
    *y = ConstEigenVectorMap<T>(a, N).dot(ConstEigenVectorMap<T>(b, N)); \
  }
CAFFE2_SPECIALIZED_DOT(float)
#undef CAFFE2_SPECIALIZED_DOT

#else // CAFFE2_USE_EIGEN_FOR_BLAS

template <>
C10_EXPORT void Gemm<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CPUContext* /*context*/,
    TensorProto::DataType /*math_type*/) {
  // MKL expects ld? >= 1
  const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
  const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
  cblas_sgemm(
      CblasRowMajor,
      trans_A,
      trans_B,
      M,
      N,
      K,
      alpha,
      A,
      lda,
      B,
      ldb,
      beta,
      C,
      N);
}

template <>
C10_EXPORT void GemmEx<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int lda,
    const float* B,
    const int ldb,
    const float beta,
    float* C,
    const int ldc,
    CPUContext* /*context*/) {
  cblas_sgemm(
      CblasRowMajor,
      trans_A,
      trans_B,
      M,
      N,
      K,
      alpha,
      A,
      lda,
      B,
      ldb,
      beta,
      C,
      ldc);
}

template <>
C10_EXPORT void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* /*context*/,
    TensorProto::DataType /*math_type*/) {
  cblas_sgemv(CblasRowMajor, trans_A, M, N, alpha, A, N, x, 1, beta, y, 1);
}

#define CAFFE2_SPECIALIZED_DOT(T, prefix)                       \
  template <>                                                   \
  C10_EXPORT void Dot<T, CPUContext>(                           \
      const int N, const T* a, const T* b, T* y, CPUContext*) { \
    *y = cblas_##prefix##dot(N, a, 1, b, 1);                    \
  }
CAFFE2_SPECIALIZED_DOT(float, s)
#undef CAFFE2_SPECIALIZED_DOT

#endif // CAFFE2_USE_EIGEN_FOR_BLAS

template <>
C10_EXPORT void GemmBatched<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float** A,
    const float** B,
    const float beta,
    float** C,
    CPUContext* context,
    TensorProto::DataType /* math_type */) {
#ifdef CAFFE2_USE_MKL
  (void)context;
  // MKL expects ld? >= 1
  const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
  const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
  const int ldc = std::max(N, 1);
  cblas_sgemm_batch(
      CblasRowMajor,
      &trans_A,
      &trans_B,
      &M,
      &N,
      &K,
      &alpha,
      A,
      &lda,
      B,
      &ldb,
      &beta,
      C,
      &ldc,
      1,
      &batch_size);
#else // CAFFE2_USE_MKL
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    math::Gemm<float, CPUContext>(
        trans_A, trans_B, M, N, K, alpha, A[i], B[i], beta, C[i], context);
  }
#endif // CAFFE2_USE_MKL
}

template <>
C10_EXPORT void GemmStridedBatched<float, CPUContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const int A_stride,
    const float* B,
    const int B_stride,
    const float beta,
    float* C,
    const int C_stride,
    CPUContext* context,
    TensorProto::DataType /* math_type */) {
#ifdef CAFFE2_USE_MKL
  (void)context;
  // MKL expects ld? >= 1
  const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
  const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);
  const int ldc = std::max(N, 1);
  std::vector<const float*> A_array(batch_size);
  std::vector<const float*> B_array(batch_size);
  std::vector<float*> C_array(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    A_array[i] = A + i * A_stride;
    B_array[i] = B + i * B_stride;
    C_array[i] = C + i * C_stride;
  }
  cblas_sgemm_batch(
      CblasRowMajor,
      &trans_A,
      &trans_B,
      &M,
      &N,
      &K,
      &alpha,
      A_array.data(),
      &lda,
      B_array.data(),
      &ldb,
      &beta,
      C_array.data(),
      &ldc,
      1,
      &batch_size);
#else // CAFFE2_USE_MKL
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    math::Gemm<float, CPUContext>(
        trans_A, trans_B, M, N, K, alpha, A, B, beta, C, context);
    A += A_stride;
    B += B_stride;
    C += C_stride;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Common math functions being used in Caffe that do not have a BLAS or MKL
// equivalent. For all these functions, we will simply implement them either via
// Eigen or via custom code.
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T>
C10_EXPORT void BroadcastImpl(
    const int X_ndim,
    const int* X_dims,
    const int Y_ndim,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    CPUContext* context) {
  CAFFE_ENFORCE_LE(X_ndim, Y_ndim);
  std::vector<int> X_dims_vector(Y_ndim);
  const int d = Y_ndim - X_ndim;
  std::fill(X_dims_vector.begin(), X_dims_vector.begin() + d, 1);
  for (int i = d; i < Y_ndim; ++i) {
    CAFFE_ENFORCE(X_dims[i - d] == 1 || X_dims[i - d] == Y_dims[i]);
    X_dims_vector[i] = X_dims[i - d];
  }
  X_dims = X_dims_vector.data();
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + Y_ndim, 1, std::multiplies<int>());
  std::vector<int> index(Y_ndim, 0);
  for (int Y_index = 0; Y_index < Y_size; ++Y_index) {
    const int X_index = utils::GetIndexFromDims(Y_ndim, X_dims, index.data());
    Y[Y_index] = X[X_index];
    utils::IncreaseIndexInDims(Y_ndim, Y_dims, index.data());
  }
  Scale<T, T, CPUContext>(Y_size, alpha, Y, Y, context);
}

} // namespace

#define CAFFE2_SPECIALIZED_BROADCAST(T)                                     \
  template <>                                                               \
  C10_EXPORT void Broadcast<T, CPUContext>(                                 \
      const int X_ndim,                                                     \
      const int* X_dims,                                                    \
      const int Y_ndim,                                                     \
      const int* Y_dims,                                                    \
      const T alpha,                                                        \
      const T* X,                                                           \
      T* Y,                                                                 \
      CPUContext* context) {                                                \
    BroadcastImpl<T>(X_ndim, X_dims, Y_ndim, Y_dims, alpha, X, Y, context); \
  }
CAFFE2_SPECIALIZED_BROADCAST(std::int32_t)
CAFFE2_SPECIALIZED_BROADCAST(std::int64_t)
CAFFE2_SPECIALIZED_BROADCAST(float)
CAFFE2_SPECIALIZED_BROADCAST(double)
#undef CAFFE2_SPECIALIZED_BROADCAST

#define CAFFE2_SPECIALIZED_INV_STD(T)                            \
  template <>                                                    \
  void InvStd<T, CPUContext>(                                    \
      const int N,                                               \
      const T epsilon,                                           \
      const T* var,                                              \
      T* inv_std,                                                \
      CPUContext* context) {                                     \
    EigenVectorArrayMap<T>(inv_std, N) =                         \
        (ConstEigenVectorArrayMap<T>(var, N) + epsilon).rsqrt(); \
  }
CAFFE2_SPECIALIZED_INV_STD(float)
#undef CAFFE2_SPECIALIZED_INV_STD

#define CAFFE2_SPECIALIZED_ROWWISEMAX(T)                         \
  template <>                                                    \
  C10_EXPORT void RowwiseMax<T, CPUContext>(                     \
      const int N, const int D, const T* x, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, N) =                                    \
        ConstEigenMatrixMap<T>(x, D, N).colwise().maxCoeff();    \
  }
CAFFE2_SPECIALIZED_ROWWISEMAX(float)
#undef CAFFE2_SPECIALIZED_ROWWISEMAX

#define CAFFE2_SPECIALIZED_COLWISEMAX(T)                         \
  template <>                                                    \
  C10_EXPORT void ColwiseMax<T, CPUContext>(                     \
      const int N, const int D, const T* x, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, D) =                                    \
        ConstEigenMatrixMap<T>(x, D, N).rowwise().maxCoeff();    \
  }
CAFFE2_SPECIALIZED_COLWISEMAX(float)
#undef CAFFE2_SPECIALIZED_COLWISEMAX

#define CAFFE2_SPECIALIZED_MAXIMUM(T)                                          \
  template <>                                                                  \
  C10_EXPORT void Maximum<T, CPUContext>(                                      \
      const int N, const float alpha, const T* x, T* y, CPUContext* context) { \
    std::transform(                                                            \
        x, x + N, y, [&alpha](const T& x_i) { return std::max(x_i, alpha); }); \
  }
CAFFE2_SPECIALIZED_MAXIMUM(float)
#undef CAFFE2_SPECIALIZED_MAXIMUM

// The actual implementation uses eigen which is column major, so notice the
// row/column swap in the actual implementation.

#define DELEGATE_EIGEN_2D_BROADCAST_1ST_BINARY_FUNCTION(T, Func, expr) \
  template <>                                                          \
  C10_EXPORT void Rowwise##Func<T, CPUContext, true>(                  \
      const int rows,                                                  \
      const int cols,                                                  \
      const T* A,                                                      \
      const T* B,                                                      \
      T* C,                                                            \
      CPUContext*) {                                                   \
    if (C == B) {                                                      \
      EigenArrayMap<T>(C, cols, rows).colwise() expr## =               \
          ConstEigenVectorArrayMap<T>(A, cols);                        \
    } else {                                                           \
      EigenArrayMap<T>(C, cols, rows) =                                \
          ConstEigenArrayMap<T>(B, cols, rows)                         \
              .colwise() expr ConstEigenVectorArrayMap<T>(A, cols);    \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  C10_EXPORT void Colwise##Func<T, CPUContext, true>(                  \
      const int rows,                                                  \
      const int cols,                                                  \
      const T* A,                                                      \
      const T* B,                                                      \
      T* C,                                                            \
      CPUContext*) {                                                   \
    if (C == B) {                                                      \
      EigenArrayMap<T>(C, cols, rows).rowwise() expr## =               \
          ConstEigenVectorArrayMap<T>(A, rows).transpose();            \
    } else {                                                           \
      EigenArrayMap<T>(C, cols, rows) =                                \
          ConstEigenArrayMap<T>(B, cols, rows)                         \
              .rowwise() expr ConstEigenVectorArrayMap<T>(A, rows)     \
              .transpose();                                            \
    }                                                                  \
  }

#define DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Func, expr) \
  template <>                                                          \
  C10_EXPORT void Rowwise##Func<T, CPUContext, false>(                 \
      const int rows,                                                  \
      const int cols,                                                  \
      const T* A,                                                      \
      const T* B,                                                      \
      T* C,                                                            \
      CPUContext*) {                                                   \
    if (C == A) {                                                      \
      EigenArrayMap<T>(C, cols, rows).colwise() expr## =               \
          ConstEigenVectorArrayMap<T>(B, cols);                        \
    } else {                                                           \
      EigenArrayMap<T>(C, cols, rows) =                                \
          ConstEigenArrayMap<T>(A, cols, rows)                         \
              .colwise() expr ConstEigenVectorArrayMap<T>(B, cols);    \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  C10_EXPORT void Colwise##Func<T, CPUContext, false>(                 \
      const int rows,                                                  \
      const int cols,                                                  \
      const T* A,                                                      \
      const T* B,                                                      \
      T* C,                                                            \
      CPUContext*) {                                                   \
    if (C == A) {                                                      \
      EigenArrayMap<T>(C, cols, rows).rowwise() expr## =               \
          ConstEigenVectorArrayMap<T>(B, rows).transpose();            \
    } else {                                                           \
      EigenArrayMap<T>(C, cols, rows) =                                \
          ConstEigenArrayMap<T>(A, cols, rows)                         \
              .rowwise() expr ConstEigenVectorArrayMap<T>(B, rows)     \
              .transpose();                                            \
    }                                                                  \
  }

#define DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(T, Func, expr) \
  DELEGATE_EIGEN_2D_BROADCAST_1ST_BINARY_FUNCTION(T, Func, expr)   \
  DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Func, expr)

#define DEFINE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(Func, expr)           \
  DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(float, Func, expr)        \
  DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(double, Func, expr)       \
  DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, Func, expr) \
  DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, Func, expr)

DEFINE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(Add, +)
DEFINE_EIGEN_2D_BROADCAST_BINARY_FUNCTION(Mul, *)

#undef DEFINE_EIGEN_2D_BROADCAST_BINARY_FUNCTION
#undef DELEGATE_EIGEN_2D_BROADCAST_BINARY_FUNCTION

#define DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION(T)           \
  template <>                                               \
  C10_EXPORT void RowwiseSub<T, CPUContext, true>(          \
      const int rows,                                       \
      const int cols,                                       \
      const T* A,                                           \
      const T* B,                                           \
      T* C,                                                 \
      CPUContext*) {                                        \
    EigenArrayMap<T>(C, cols, rows) =                       \
        (-ConstEigenArrayMap<T>(B, cols, rows)).colwise() + \
        ConstEigenVectorArrayMap<T>(A, cols);               \
  }                                                         \
  template <>                                               \
  C10_EXPORT void ColwiseSub<T, CPUContext, true>(          \
      const int rows,                                       \
      const int cols,                                       \
      const T* A,                                           \
      const T* B,                                           \
      T* C,                                                 \
      CPUContext*) {                                        \
    EigenArrayMap<T>(C, cols, rows) =                       \
        (-ConstEigenArrayMap<T>(B, cols, rows)).rowwise() + \
        ConstEigenVectorArrayMap<T>(A, rows).transpose();   \
  }                                                         \
  DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Sub, -)

DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION(float)
DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION(double)
DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION(std::int32_t)
DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION(std::int64_t)

#undef DEFINE_EIGEN_2D_BROADCAST_SUB_FUNCTION

#define DEFINE_EIGEN_2D_BROADCAST_DIV_FUNCTION(T)                  \
  template <>                                                      \
  C10_EXPORT void RowwiseDiv<T, CPUContext, true>(                 \
      const int rows,                                              \
      const int cols,                                              \
      const T* A,                                                  \
      const T* B,                                                  \
      T* C,                                                        \
      CPUContext*) {                                               \
    EigenArrayMap<T>(C, cols, rows) =                              \
        ConstEigenArrayMap<T>(B, cols, rows).inverse().colwise() * \
        ConstEigenVectorArrayMap<T>(A, cols);                      \
  }                                                                \
  template <>                                                      \
  C10_EXPORT void ColwiseDiv<T, CPUContext, true>(                 \
      const int rows,                                              \
      const int cols,                                              \
      const T* A,                                                  \
      const T* B,                                                  \
      T* C,                                                        \
      CPUContext*) {                                               \
    EigenArrayMap<T>(C, cols, rows) =                              \
        ConstEigenArrayMap<T>(B, cols, rows).inverse().rowwise() * \
        ConstEigenVectorArrayMap<T>(A, rows).transpose();          \
  }                                                                \
  DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(T, Div, /)

DEFINE_EIGEN_2D_BROADCAST_DIV_FUNCTION(float)
DEFINE_EIGEN_2D_BROADCAST_DIV_FUNCTION(double)
DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(std::int32_t, Div, /)
DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION(std::int64_t, Div, /)

#undef DEFINE_EIGEN_2D_BROADCAST_DIV_FUNCTION

#undef DELEGATE_EIGEN_2D_BROADCAST_1ST_BINARY_FUNCTION
#undef DELEGATE_EIGEN_2D_BROADCAST_2ND_BINARY_FUNCTION

template <>
C10_EXPORT void Not<bool, CPUContext>(
    const int N,
    const bool* x,
    bool* y,
    CPUContext* /*context*/) {
  for (int i = 0; i < N; ++i) {
    y[i] = !x[i];
  }
}

#undef C10_DEFINE_BINARY_OP
#undef CAFFE2_INSTANTIATE_BINARY_OP

#define CAFFE2_SPECIALIZED_CPU_ADD_STRIPED_BATCH(T)             \
  template <>                                                   \
  C10_EXPORT void AddStripedBatch(                              \
      const int N,                                              \
      const T* first,                                           \
      T* y,                                                     \
      const int stripe,                                         \
      const int batch,                                          \
      CPUContext* context) {                                    \
    for (int j = 0; j < batch; j++) {                           \
      Add<T, CPUContext>(N, first + j * stripe, y, y, context); \
    }                                                           \
  }

CAFFE2_SPECIALIZED_CPU_ADD_STRIPED_BATCH(float);
#undef CAFFE2_SPECIALIZED_CPU_ADD_STRIPED_BATCH

namespace {

template <typename TIn, typename TOut, class BinaryOperator, bool kBroadcast1st>
C10_EXPORT void RowwiseBinaryOp(
    const int rows,
    const int cols,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int C_index = i * cols + j;
      const int A_index = kBroadcast1st ? j : C_index;
      const int B_index = kBroadcast1st ? C_index : j;
      C[C_index] = op(A[A_index], B[B_index]);
    }
  }
}

template <typename TIn, typename TOut, class BinaryOperator, bool kBroadcast1st>
C10_EXPORT void ColwiseBinaryOp(
    const int rows,
    const int cols,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int C_index = i * cols + j;
      const int A_index = kBroadcast1st ? i : C_index;
      const int B_index = kBroadcast1st ? C_index : i;
      C[C_index] = op(A[A_index], B[B_index]);
    }
  }
}

template <typename TIn, typename TOut, class BinaryOperator>
C10_EXPORT void BroadcastBinaryOpImpl(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const BinaryOperator& op,
    const TIn* A,
    const TIn* B,
    TOut* C) {
  std::vector<int> index(ndim, 0);
  const int C_size =
      std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
  for (int C_index = 0; C_index < C_size; ++C_index) {
    const int A_index = utils::GetIndexFromDims(ndim, A_dims, index.data());
    const int B_index = utils::GetIndexFromDims(ndim, B_dims, index.data());
    C[C_index] = op(A[A_index], B[B_index]);
    utils::IncreaseIndexInDims(ndim, C_dims, index.data());
  }
}

} // namespace

#define DELEGATE_2D_BROADCAST_BINARY_FUNCTION(TIn, TOut, Func, Op)             \
  template <>                                                                  \
  C10_EXPORT void Rowwise##Func<TIn, CPUContext, true>(                        \
      const int rows,                                                          \
      const int cols,                                                          \
      const TIn* A,                                                            \
      const TIn* B,                                                            \
      TOut* C,                                                                 \
      CPUContext*) {                                                           \
    RowwiseBinaryOp<TIn, TOut, Op<TIn>, true>(rows, cols, Op<TIn>(), A, B, C); \
  }                                                                            \
  template <>                                                                  \
  C10_EXPORT void Rowwise##Func<TIn, CPUContext, false>(                       \
      const int rows,                                                          \
      const int cols,                                                          \
      const TIn* A,                                                            \
      const TIn* B,                                                            \
      TOut* C,                                                                 \
      CPUContext*) {                                                           \
    RowwiseBinaryOp<TIn, TOut, Op<TIn>, false>(                                \
        rows, cols, Op<TIn>(), A, B, C);                                       \
  }                                                                            \
  template <>                                                                  \
  C10_EXPORT void Colwise##Func<TIn, CPUContext, true>(                        \
      const int rows,                                                          \
      const int cols,                                                          \
      const TIn* A,                                                            \
      const TIn* B,                                                            \
      TOut* C,                                                                 \
      CPUContext*) {                                                           \
    ColwiseBinaryOp<TIn, TOut, Op<TIn>, true>(rows, cols, Op<TIn>(), A, B, C); \
  }                                                                            \
  template <>                                                                  \
  C10_EXPORT void Colwise##Func<TIn, CPUContext, false>(                       \
      const int rows,                                                          \
      const int cols,                                                          \
      const TIn* A,                                                            \
      const TIn* B,                                                            \
      TOut* C,                                                                 \
      CPUContext*) {                                                           \
    ColwiseBinaryOp<TIn, TOut, Op<TIn>, false>(                                \
        rows, cols, Op<TIn>(), A, B, C);                                       \
  }

#define DEFINE_2D_COMPARE_FUNCTION(Func, Op)                          \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_2D_COMPARE_FUNCTION(EQ, std::equal_to)
DEFINE_2D_COMPARE_FUNCTION(NE, std::not_equal_to)
DEFINE_2D_COMPARE_FUNCTION(LT, std::less)
DEFINE_2D_COMPARE_FUNCTION(LE, std::less_equal)
DEFINE_2D_COMPARE_FUNCTION(GT, std::greater)
DEFINE_2D_COMPARE_FUNCTION(GE, std::greater_equal)

#undef DEFINE_2D_COMPARE_FUNCTION

DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, And, std::logical_and)
DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Or, std::logical_or)
DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Xor, std::bit_xor)

#define DEFINE_2D_BROADCAST_BITWISE_BINARY_FUNCTION(Func, Op)                 \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)                 \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_2D_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_2D_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseAnd, std::bit_and)
DEFINE_2D_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseOr, std::bit_or)
DEFINE_2D_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseXor, std::bit_xor)

#undef DEFINE_2D_BROADCAST_BITWISE_BINARY_FUNCTION

#undef DELEGATE_2D_BROADCAST_BINARY_FUNCTION

#define DEFINE_2D_BROADCAST_1ST_DIV_FUNCTION(T)    \
  template <>                                      \
  C10_EXPORT void RowwiseDiv<T, CPUContext, true>( \
      const int rows,                              \
      const int cols,                              \
      const T* A,                                  \
      const T* B,                                  \
      T* C,                                        \
      CPUContext*) {                               \
    RowwiseBinaryOp<T, T, std::divides<T>, true>(  \
        rows, cols, std::divides<T>(), A, B, C);   \
  }                                                \
  template <>                                      \
  C10_EXPORT void ColwiseDiv<T, CPUContext, true>( \
      const int rows,                              \
      const int cols,                              \
      const T* A,                                  \
      const T* B,                                  \
      T* C,                                        \
      CPUContext*) {                               \
    ColwiseBinaryOp<T, T, std::divides<T>, true>(  \
        rows, cols, std::divides<T>(), A, B, C);   \
  }
DEFINE_2D_BROADCAST_1ST_DIV_FUNCTION(std::int32_t)
DEFINE_2D_BROADCAST_1ST_DIV_FUNCTION(std::int64_t)
#undef DEFINE_2D_BROADCAST_1ST_DIV_FUNCTION

#define DELEGATE_BROADCAST_BINARY_FUNCTION(TIn, TOut, Func, Op)              \
  template <>                                                                \
  C10_EXPORT void Func<TIn, CPUContext>(                                     \
      const int A_ndim,                                                      \
      const int* A_dims,                                                     \
      const int B_ndim,                                                      \
      const int* B_dims,                                                     \
      const TIn* A,                                                          \
      const TIn* B,                                                          \
      TOut* C,                                                               \
      CPUContext* context) {                                                 \
    const int ndim = std::max(A_ndim, B_ndim);                               \
    std::vector<int> A_dims_array(ndim);                                     \
    std::vector<int> B_dims_array(ndim);                                     \
    std::vector<int> C_dims_array(ndim);                                     \
    utils::ComputeBroadcastBinaryOpDims(                                     \
        A_ndim,                                                              \
        A_dims,                                                              \
        B_ndim,                                                              \
        B_dims,                                                              \
        A_dims_array.data(),                                                 \
        B_dims_array.data(),                                                 \
        C_dims_array.data());                                                \
    if (A_dims_array == B_dims_array) {                                      \
      const int size = std::accumulate(                                      \
          C_dims_array.cbegin(),                                             \
          C_dims_array.cend(),                                               \
          1,                                                                 \
          std::multiplies<int>());                                           \
      Func<TIn, CPUContext>(size, A, B, C, context);                         \
      return;                                                                \
    }                                                                        \
    int rows;                                                                \
    int cols;                                                                \
    bool broadcast_1st;                                                      \
    if (utils::IsRowwiseBroadcastBinaryOp(                                   \
            ndim,                                                            \
            A_dims_array.data(),                                             \
            B_dims_array.data(),                                             \
            &rows,                                                           \
            &cols,                                                           \
            &broadcast_1st)) {                                               \
      if (broadcast_1st) {                                                   \
        Rowwise##Func<TIn, CPUContext, true>(rows, cols, A, B, C, context);  \
      } else {                                                               \
        Rowwise##Func<TIn, CPUContext, false>(rows, cols, A, B, C, context); \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    if (utils::IsColwiseBroadcastBinaryOp(                                   \
            ndim,                                                            \
            A_dims_array.data(),                                             \
            B_dims_array.data(),                                             \
            &rows,                                                           \
            &cols,                                                           \
            &broadcast_1st)) {                                               \
      if (broadcast_1st) {                                                   \
        Colwise##Func<TIn, CPUContext, true>(rows, cols, A, B, C, context);  \
      } else {                                                               \
        Colwise##Func<TIn, CPUContext, false>(rows, cols, A, B, C, context); \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    int pre;                                                                 \
    int mid;                                                                 \
    int nxt;                                                                 \
    if (utils::IsBothEndsBroadcastBinaryOp(                                  \
            ndim,                                                            \
            A_dims_array.data(),                                             \
            B_dims_array.data(),                                             \
            &pre,                                                            \
            &mid,                                                            \
            &nxt,                                                            \
            &broadcast_1st)) {                                               \
      const int stride = mid * nxt;                                          \
      for (int i = 0; i < pre; ++i) {                                        \
        if (broadcast_1st) {                                                 \
          Colwise##Func<TIn, CPUContext, true>(                              \
              mid, nxt, A, B + i * stride, C + i * stride, context);         \
        } else {                                                             \
          Colwise##Func<TIn, CPUContext, false>(                             \
              mid, nxt, A + i * stride, B, C + i * stride, context);         \
        }                                                                    \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    BroadcastBinaryOpImpl(                                                   \
        ndim,                                                                \
        A_dims_array.data(),                                                 \
        B_dims_array.data(),                                                 \
        C_dims_array.data(),                                                 \
        Op<TIn>(),                                                           \
        A,                                                                   \
        B,                                                                   \
        C);                                                                  \
  }

#define DEFINE_BROADCAST_COMPARE_FUNCTION(Func, Op)                \
  DELEGATE_BROADCAST_BINARY_FUNCTION(float, bool, Func, Op)        \
  DELEGATE_BROADCAST_BINARY_FUNCTION(double, bool, Func, Op)       \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, bool, Func, Op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, bool, Func, Op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)

DEFINE_BROADCAST_COMPARE_FUNCTION(EQ, std::equal_to)
DEFINE_BROADCAST_COMPARE_FUNCTION(NE, std::not_equal_to)
DEFINE_BROADCAST_COMPARE_FUNCTION(LT, std::less)
DEFINE_BROADCAST_COMPARE_FUNCTION(LE, std::less_equal)
DEFINE_BROADCAST_COMPARE_FUNCTION(GT, std::greater)
DEFINE_BROADCAST_COMPARE_FUNCTION(GE, std::greater_equal)

#undef DEFINE_BROADCAST_COMPARE_FUNCTION

#define DEFINE_BROADCAST_BINARY_FUNCTION(Func, Op)                         \
  DELEGATE_BROADCAST_BINARY_FUNCTION(float, float, Func, Op)               \
  DELEGATE_BROADCAST_BINARY_FUNCTION(double, double, Func, Op)             \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_BROADCAST_BINARY_FUNCTION(Add, std::plus)
DEFINE_BROADCAST_BINARY_FUNCTION(Sub, std::minus)
DEFINE_BROADCAST_BINARY_FUNCTION(Mul, std::multiplies)
DEFINE_BROADCAST_BINARY_FUNCTION(Div, std::divides)

#undef DEFINE_BROADCAST_BINARY_FUNCTION

DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, And, std::logical_and)
DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Or, std::logical_or)
DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Xor, std::bit_xor)

#define DEFINE_BROADCAST_BITWISE_BINARY_FUNCTION(Func, Op)                 \
  DELEGATE_BROADCAST_BINARY_FUNCTION(bool, bool, Func, Op)                 \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int32_t, std::int32_t, Func, Op) \
  DELEGATE_BROADCAST_BINARY_FUNCTION(std::int64_t, std::int64_t, Func, Op)

DEFINE_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseAnd, std::bit_and)
DEFINE_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseOr, std::bit_or)
DEFINE_BROADCAST_BITWISE_BINARY_FUNCTION(BitwiseXor, std::bit_xor)

#undef DEFINE_BITWISE_BROADCAST_BINARY_FUNCTION

#undef DELEGATE_BROADCAST_BINARY_FUNCTION

#define CAFFE2_RAND_UNIFORM_REAL(T)                                      \
  template <>                                                            \
  C10_EXPORT void RandUniform<T, CPUContext>(                            \
      const size_t n, const T a, const T b, T* r, CPUContext* context) { \
    std::uniform_real_distribution<T> distribution(a, b);                \
    for (size_t i = 0; i < n; ++i) {                                     \
      r[i] = distribution(context->RandGenerator());                     \
    }                                                                    \
  }
CAFFE2_RAND_UNIFORM_REAL(float);
CAFFE2_RAND_UNIFORM_REAL(double);
#undef CAFFE2_RAND_UNIFORM_REAL

#define CAFFE2_RAND_UNIFORM_CHAR(T)                                        \
  template <>                                                              \
  C10_EXPORT void RandUniform<T, CPUContext>(                              \
      const size_t n, const T a, const T b, T* r, CPUContext* context) {   \
    std::uniform_int_distribution<short> distribution((short)a, (short)b); \
    for (size_t i = 0; i < n; ++i) {                                       \
      r[i] = static_cast<T>(distribution(context->RandGenerator()));       \
    }                                                                      \
  }
CAFFE2_RAND_UNIFORM_CHAR(int8_t);
CAFFE2_RAND_UNIFORM_CHAR(uint8_t);
#undef CAFFE2_RAND_UNIFORM_CHAR

#define CAFFE2_RAND_UNIFORM_INT(T)                                       \
  template <>                                                            \
  C10_EXPORT void RandUniform<T, CPUContext>(                            \
      const size_t n, const T a, const T b, T* r, CPUContext* context) { \
    std::uniform_int_distribution<T> distribution(a, b);                 \
    for (size_t i = 0; i < n; ++i) {                                     \
      r[i] = distribution(context->RandGenerator());                     \
    }                                                                    \
  }

CAFFE2_RAND_UNIFORM_INT(int16_t);
CAFFE2_RAND_UNIFORM_INT(int32_t);
CAFFE2_RAND_UNIFORM_INT(int64_t);
CAFFE2_RAND_UNIFORM_INT(uint16_t);
CAFFE2_RAND_UNIFORM_INT(uint32_t);
CAFFE2_RAND_UNIFORM_INT(uint64_t);
#undef CAFFE2_RAND_UNIFORM_INT

// This is not uniformly distributed between a and b.
// It takes advantage of normal distribution to generate numbers
// with mean = sum / n.
// Ideally the algorithm should be generating n numbers between 0 and 1,
// sum them up as scaled_sum, and use sum / scaled_sum to adjust the values
// to between a and b.
// The algorithm is non-trivial given the adjustment would be different towards
// each value.
#define CAFFE2_RAND_FIXED_SUM(T)                                          \
  template <>                                                             \
  C10_EXPORT void RandFixedSum<T, CPUContext>(                            \
      const size_t n,                                                     \
      const T a,                                                          \
      const T b,                                                          \
      const T sum,                                                        \
      T* r,                                                               \
      CPUContext* context) {                                              \
    CAFFE_ENFORCE_GE(a, 0);                                               \
    CAFFE_ENFORCE_GE(sum / (double)n, a);                                 \
    CAFFE_ENFORCE_LE(sum / (double)n, b);                                 \
    T current_sum = 0;                                                    \
    T remaining_sum = sum;                                                \
    for (size_t i = 0; i < n; ++i) {                                      \
      auto remaining_numbers = n - 1 - i;                                 \
      double mean = (sum - current_sum) / (remaining_numbers + 1);        \
      double stdev = std::min(mean - a, b - mean);                        \
      std::normal_distribution<double> distribution{mean, stdev / 4.0};   \
      T value, remaining_sum_test;                                        \
      do {                                                                \
        value = distribution(context->RandGenerator());                   \
        remaining_sum_test = remaining_sum - value;                       \
      } while (value < a || remaining_sum_test < a * remaining_numbers || \
               value > b || remaining_sum_test > b * remaining_numbers);  \
      r[i] = value;                                                       \
      CAFFE_ENFORCE(a <= value && value <= b);                            \
      current_sum += value;                                               \
      remaining_sum -= value;                                             \
      CAFFE_ENFORCE_GE(remaining_sum, a* remaining_numbers);              \
      CAFFE_ENFORCE_LE(remaining_sum, b* remaining_numbers);              \
    }                                                                     \
    r[n - 1] += remaining_sum;                                            \
    current_sum += remaining_sum;                                         \
    CAFFE_ENFORCE(a <= r[n - 1] && r[n - 1] <= b);                        \
    CAFFE_ENFORCE_EQ(current_sum, sum);                                   \
  }
CAFFE2_RAND_FIXED_SUM(float);
CAFFE2_RAND_FIXED_SUM(double);
CAFFE2_RAND_FIXED_SUM(int8_t);
CAFFE2_RAND_FIXED_SUM(int16_t);
CAFFE2_RAND_FIXED_SUM(int32_t);
CAFFE2_RAND_FIXED_SUM(int64_t);
CAFFE2_RAND_FIXED_SUM(uint8_t);
CAFFE2_RAND_FIXED_SUM(uint16_t);
CAFFE2_RAND_FIXED_SUM(uint32_t);
CAFFE2_RAND_FIXED_SUM(uint64_t);
#undef CAFFE2_RAND_FIXED_SUM

template <class Type, class Val_t, class Ind_t, class Context_t, bool cdf_app>
Ind_t generate_stack_distance(
    std::vector<Ind_t>& cum_val,
    std::vector<Val_t>& cum_dis,
    std::vector<Ind_t>& cum_map,
    Ind_t max_i,
    Ind_t i,
    Context_t* context) {
  /* Description:
     Inverse Transform Sampling method to generate values for random variable X
     that is described by the cumulative distribution F (cum_val,cum_dis).
     Notice, that we may choose to use the inverse map of F (cum_map) as an
     approximation to avoid searching. Also, scaling the probability so that
     the values are within max_i refs, because stack distance can not be >
     than the # of already generated refs (max_i).
  */
  Ind_t j, k, n;
  Val_t u, f, fi;

  // generate a random number u in [0,1] from a uniform distribution U
  math::RandUniform<Val_t, Context_t>(1, 0, 1, &u, context);

  // scale the random number u to be within range [0,f(i)], if needed
  if (i < max_i) {
    // approach 2: allows gaps in the distribution
    j = (std::upper_bound(cum_val.begin(), cum_val.end(), i) -
         cum_val.begin()) -
        1;
    fi = cum_dis[j];
    u *= fi;
  }
  // 2. compute the stack distance value of x, s.t. F(x)=u
  // notice that the cumulative distribution F increases monotonically up to 1
  if (cdf_app) {
    // look up cum_val corresponding to u <= cum_dis[j]
    k = cum_map.size();
    n = (Ind_t)round(u * k);
    j = cum_map[n];
    return cum_val[j];
  } else {
    // iterate until you find the cum_val corresponding to u <= cum_dis[j]
    for (j = 0; j < Ind_t(cum_dis.size()); j++) {
      f = cum_dis[j];
      if (u <= f) {
        return cum_val[j];
      }
    }
    return cum_val[j - 1];
  }
}

template <class Type, class Val_t, class Ind_t, class Context_t, bool cdf_app>
C10_EXPORT void generate_trace_lru(
    std::vector<Ind_t>& uni_ref,
    std::vector<Ind_t>& cum_val,
    std::vector<Val_t>& cum_dis,
    std::vector<Ind_t>& cum_map,
    Context_t* context,
    Ind_t cache_line_size,
    Ind_t n,
    Type min,
    Type max,
    Type* syn_ref) {
  /* Description:
     Generate synthetic trace from a list of unique accesses uni_ref, and
     cumulative distribution of distances (cum_val,cum_dis) between them.
     Also, there is an option to use cum_map approximation to avoid searching.
  */
  Ind_t i, j, k, sd, line_ref, mem_ref, mem_ref_within_line;
  Ind_t max_sd = cum_val.back();
  Ind_t l = uni_ref.size();

  for (i = 0, j = 0; j < n; j++) {
    // generate stack distance
    sd = generate_stack_distance<Type, Val_t, Ind_t, Context_t, cdf_app>(
        cum_val, cum_dis, cum_map, max_sd, i, context);
    // fixed access within cache line
    mem_ref_within_line = 0;
    // random access within cache line
    // Val_t r;
    // math::RandUniform<Val_t, Context_t>(1, 0, 1, &r, context);
    // mem_ref_within_line = floor(r*cache_line_size);

    // generate memory reference
    if (sd == 0) {
      k = 0; /// new reference ///
      i++;
    } else {
      k = l - sd; /// existing reference ///
    }
    line_ref = uni_ref[k]; // pop k-th element
    uni_ref.erase(uni_ref.begin() + k);
    uni_ref.push_back(line_ref); // append it back
    mem_ref = line_ref * cache_line_size + mem_ref_within_line;
    /*
    //debug prints
    if ((mem_ref < min) || (mem_ref > max)) {
      //printf("mem_ref[%d]=%d (%ld) \n",j,mem_ref,syn_ref[j]);
      std::cout << "syn_ref[" << j << "]=" << (Type)mem_ref << " ";
      std::cout << "(" << mem_ref << ") ";
      std::cout << "[" << min << "," << max << "]" << std::endl;
      int scanf_temp;
      scanf("%d",&scanf_temp);
    }
    */

    // patch mem_ref to be within range
    // WARNING: this should not be needed if instantiation type and distribution
    // choice is correct. It is remeding a symptom of earlier mistakes.
    if (mem_ref < min) {
      mem_ref = min;
      // std::cout << "clamping (min) mem_ref=" << mem_ref << std::endl;
    }
    if (mem_ref > max) {
      mem_ref = max; // mem_ref % max;
      // std::cout << "clamping (max) mem_ref=" << mem_ref << std::endl;
    }

    // save generated memory reference
    syn_ref[j] = (Type)mem_ref;
  }
}

// Generate n values from synthetic data distribution,
// define by unique accesses and stack distances
// WARNING: can create this for all tables or per table, but in latter
// case we need to know the table id, to sample from the right distribution
#define CAFFE2_RAND_SYNTHETIC_DATA(T)                                         \
  template <>                                                                 \
  C10_EXPORT void RandSyntheticData<T, CPUContext>(                           \
      const size_t n, const T a, const T b, T* r, CPUContext* context) {      \
    /* unique memory references */                                            \
    std::vector<int> mem_ref = {1, 2, 3, 4, 5, 6};                            \
    /* cumulative distribution of distances */                                \
    std::vector<int> cum_val = {0, 1, 3, 4, 5};                               \
    std::vector<double> cum_dis = {0.55, 0.64, 0.82, 0.91, 1.0};              \
    /* inverse map of cumulative distribution (for O(1) lookup) */            \
    /* std::vector<int> cum_map = {0, 0, 0, 0, 0, 1, 2, 2, 3, 4}; */          \
    int k = 10; /* 100; */                                                    \
    std::vector<int> cum_map(k, 0);                                           \
    for (int j = 0; j < cum_dis.size();) {                                    \
      int sz = (int)round(cum_dis[j] * k);                                    \
      for (int i = 0; i < sz; i++) {                                          \
        cum_map[j + i] = j;                                                   \
      }                                                                       \
      j += sz;                                                                \
    }                                                                         \
                                                                              \
    /* code to generate the synthetic data from the above values */           \
    const int cache_line = 1; /* 64; */                                       \
    generate_trace_lru<T, double, int, CPUContext, false>(                    \
        mem_ref, cum_val, cum_dis, cum_map, context, cache_line, n, a, b, r); \
  }

CAFFE2_RAND_SYNTHETIC_DATA(float);
CAFFE2_RAND_SYNTHETIC_DATA(double);
CAFFE2_RAND_SYNTHETIC_DATA(int8_t);
CAFFE2_RAND_SYNTHETIC_DATA(int16_t);
CAFFE2_RAND_SYNTHETIC_DATA(int32_t);
CAFFE2_RAND_SYNTHETIC_DATA(int64_t);
CAFFE2_RAND_SYNTHETIC_DATA(uint8_t);
CAFFE2_RAND_SYNTHETIC_DATA(uint16_t);
CAFFE2_RAND_SYNTHETIC_DATA(uint32_t);
CAFFE2_RAND_SYNTHETIC_DATA(uint64_t);
#undef CAFFE2_RAND_SYNTHETIC_DATA

#define CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(T)                    \
  template <>                                                        \
  C10_EXPORT void RandUniformUnique<T, CPUContext>(                  \
      const size_t n,                                                \
      const T a,                                                     \
      const T b,                                                     \
      T* r,                                                          \
      const size_t m,                                                \
      const T* avoid,                                                \
      CPUContext* context) {                                         \
    CAFFE_ENFORCE_LE(                                                \
        n, b - a - m + 1, "Cannot satisfy the unique requirement");  \
    std::unordered_set<T> avoid_set(n);                              \
    if (m) {                                                         \
      avoid_set.insert(avoid, avoid + m);                            \
      CAFFE_ENFORCE_EQ(                                              \
          m, avoid_set.size(), "AC10_EXPORT void should be unique"); \
    }                                                                \
    std::uniform_int_distribution<T> distribution(a, b);             \
    T v = 0;                                                         \
    for (size_t i = 0; i < n; ++i) {                                 \
      do {                                                           \
        v = distribution(context->RandGenerator());                  \
      } while (avoid_set.count(v));                                  \
      r[i] = v;                                                      \
      avoid_set.insert(v);                                           \
    }                                                                \
  }

CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(int32_t);
CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(int64_t);
#undef CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE

template <>
C10_EXPORT void RandGaussian<float, CPUContext>(
    const size_t n,
    const float mean,
    const float std,
    float* r,
    CPUContext* context) {
  std::normal_distribution<float> distribution(mean, std);
  for (size_t i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

#define CAFFE2_SPECIALIZED_SUM(T)            \
  template <>                                \
  C10_EXPORT void Sum<T, CPUContext>(        \
      const int N,                           \
      const T* x,                            \
      T* y,                                  \
      CPUContext* /* unused */,              \
      Tensor* /* unused */) {                \
    *y = ConstEigenVectorMap<T>(x, N).sum(); \
  }

CAFFE2_SPECIALIZED_SUM(float);
CAFFE2_SPECIALIZED_SUM(int32_t);
CAFFE2_SPECIALIZED_SUM(int64_t);

#undef CAFFE2_SPECIALIZED_SUM

template <>
C10_EXPORT void SumSqr<float, CPUContext>(
    const int N,
    const float* x,
    float* y,
    CPUContext* /*context*/ /* unused */,
    Tensor* /*scratch_ptr*/ /* unused */) {
  *y = ConstEigenVectorMap<float>(x, N).squaredNorm();
}

template <>
C10_EXPORT void Select<float, CPUContext>(
    const int N,
    const int D,
    const float* x,
    const int* idx,
    float* y,
    CPUContext* /*context*/) {
  for (int i = 0; i < N; ++i) {
    DCHECK_LT(idx[i], D);
    y[i] = x[i * D + idx[i]];
  }
}

template <>
C10_EXPORT void CopyMatrix<CPUContext>(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    CPUContext* /*context*/,
    TypeMeta::Copy copy) {
  if (A == nullptr || B == nullptr) {
    return;
  }
  if (lda == N && ldb == N) {
    // can coalesce to a single memcpy of size M * N
    if (copy) {
      copy(static_cast<const char*>(A), static_cast<char*>(B), N * M);
    } else {
      memcpy(
          static_cast<char*>(B), static_cast<const char*>(A), itemsize * N * M);
    }
    return;
  }

  for (int i = 0; i < M; ++i) {
    if (copy) {
      copy(
          static_cast<const char*>(A) + lda * i * itemsize,
          static_cast<char*>(B) + ldb * i * itemsize,
          N);
    } else {
      memcpy(
          static_cast<char*>(B) + ldb * i * itemsize,
          static_cast<const char*>(A) + lda * i * itemsize,
          itemsize * N);
    }
  }
}

#ifdef CAFFE2_USE_MKL

#define DELEGATE_COPY_MATRIX_FUNCTION(T, Func)  \
  template <>                                   \
  C10_EXPORT void CopyMatrix<T, CPUContext>(    \
      const int M,                              \
      const int N,                              \
      const T* A,                               \
      const int lda,                            \
      T* B,                                     \
      const int ldb,                            \
      CPUContext* /* context */) {              \
    Func('R', 'N', M, N, T(1), A, lda, B, ldb); \
  }                                             \
  template <>                                   \
  C10_EXPORT void CopyMatrix<T, CPUContext>(    \
      const int M,                              \
      const int N,                              \
      const T* A,                               \
      const int A_outer_stride,                 \
      const int A_inner_stride,                 \
      T* B,                                     \
      const int B_outer_stride,                 \
      const int B_inner_stride,                 \
      CPUContext* /* context */) {              \
    Func##2(                                    \
        'R',                                    \
        'N',                                    \
        M,                                      \
        N,                                      \
        T(1),                                   \
        A,                                      \
        A_outer_stride,                         \
        A_inner_stride,                         \
        B,                                      \
        B_outer_stride,                         \
        B_inner_stride);                        \
  }
DELEGATE_COPY_MATRIX_FUNCTION(float, mkl_somatcopy)
DELEGATE_COPY_MATRIX_FUNCTION(double, mkl_domatcopy)
#undef DELEGATE_COPY_MATRIX_FUNCTION

#endif // CAFFE2_USE_MKL

#define CAFFE2_SPECIALIZED_COPY_MATRIX(T)                                \
  template <>                                                            \
  C10_EXPORT void CopyMatrix<T, CPUContext>(                             \
      const int M,                                                       \
      const int N,                                                       \
      const T* A,                                                        \
      const int lda,                                                     \
      T* B,                                                              \
      const int ldb,                                                     \
      CPUContext* /* context */) {                                       \
    if (M == 0 || N == 0) {                                              \
      return;                                                            \
    }                                                                    \
    if (lda == N) {                                                      \
      if (ldb == N) {                                                    \
        std::memcpy(B, A, sizeof(T) * M * N);                            \
      } else {                                                           \
        EigenOuterStridedMatrixMap<T>(B, N, M, EigenOuterStride(ldb)) =  \
            ConstEigenMatrixMap<T>(A, N, M);                             \
      }                                                                  \
    } else {                                                             \
      if (ldb == N) {                                                    \
        EigenMatrixMap<T>(B, N, M) = ConstEigenOuterStridedMatrixMap<T>( \
            A, N, M, EigenOuterStride(lda));                             \
      } else {                                                           \
        EigenOuterStridedMatrixMap<T>(B, N, M, EigenOuterStride(ldb)) =  \
            ConstEigenOuterStridedMatrixMap<T>(                          \
                A, N, M, EigenOuterStride(lda));                         \
      }                                                                  \
    }                                                                    \
  }                                                                      \
  template <>                                                            \
  C10_EXPORT void CopyMatrix<T, CPUContext>(                             \
      const int M,                                                       \
      const int N,                                                       \
      const T* A,                                                        \
      const int A_outer_stride,                                          \
      const int A_inner_stride,                                          \
      T* B,                                                              \
      const int B_outer_stride,                                          \
      const int B_inner_stride,                                          \
      CPUContext* context) {                                             \
    if (A_inner_stride == 1 && B_inner_stride == 1) {                    \
      CopyMatrix<T, CPUContext>(                                         \
          M, N, A, A_outer_stride, B, B_outer_stride, context);          \
      return;                                                            \
    }                                                                    \
    EigenStridedMatrixMap<T>(                                            \
        B, N, M, EigenStride(B_outer_stride, B_inner_stride)) =          \
        ConstEigenStridedMatrixMap<T>(                                   \
            A, N, M, EigenStride(A_outer_stride, A_inner_stride));       \
  }

#ifndef CAFFE2_USE_MKL
CAFFE2_SPECIALIZED_COPY_MATRIX(float)
CAFFE2_SPECIALIZED_COPY_MATRIX(double)
#endif // CAFFE2_USE_MKL

CAFFE2_SPECIALIZED_COPY_MATRIX(int)
CAFFE2_SPECIALIZED_COPY_MATRIX(int64_t)
CAFFE2_SPECIALIZED_COPY_MATRIX(std::uint8_t)
CAFFE2_SPECIALIZED_COPY_MATRIX(std::uint16_t)

#undef CAFFE2_SPECIALIZXED_COPY_MATRIX

namespace {

template <typename T>
C10_EXPORT void Im2ColZeroPaddingAndNoDilationNCHW(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const T* img_data,
    T* col_data,
    CPUContext* context) {
  const int output_h = (H - kernel_h) / stride_h + 1;
  const int output_w = (W - kernel_w) / stride_w + 1;
  const int output_size = output_h * output_w;
  for (int c = 0; c < C; ++c) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const T* src = img_data + kh * W + kw;
        if (stride_w == 1) {
          CopyMatrix<T, CPUContext>(
              output_h,
              output_w,
              src,
              stride_h * W,
              col_data,
              output_w,
              context);
        } else {
          CopyMatrix<T, CPUContext>(
              output_h,
              output_w,
              src,
              stride_h * W,
              stride_w,
              col_data,
              output_w,
              1,
              context);
        }
        col_data += output_size;
      }
    }
    img_data += H * W;
  }
}

template <typename T>
C10_EXPORT void Col2ImZeroPaddingAndNoDilationNCHW(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const T* col_data,
    T* img_data,
    CPUContext* context) {
  Set<T, CPUContext>(C * H * W, T(0), img_data, context);
  const int output_h = (H - kernel_h) / stride_h + 1;
  const int output_w = (W - kernel_w) / stride_w + 1;
  const int output_size = output_h * output_w;
  for (int c = 0; c < C; ++c) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        T* dst = img_data + kh * W + kw;
        if (stride_w == 1) {
          EigenOuterStridedArrayMap<T>(
              dst, output_w, output_h, EigenOuterStride(stride_h * W)) +=
              ConstEigenArrayMap<T>(col_data, output_w, output_h);
        } else {
          EigenStridedArrayMap<T>(
              dst, output_w, output_h, EigenStride(stride_h * W, stride_w)) +=
              ConstEigenArrayMap<T>(col_data, output_w, output_h);
        }
        col_data += output_size;
      }
    }
    img_data += H * W;
  }
}

template <typename T>
C10_EXPORT void Im2ColZeroPaddingAndNoDilationNHWC(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const T* img_data,
    T* col_data,
    CPUContext* context) {
  const int output_h = (H - kernel_h) / stride_h + 1;
  const int output_w = (W - kernel_w) / stride_w + 1;
  const int kernel_size = kernel_h * kernel_w;
  for (int yh = 0; yh < output_h; ++yh) {
    for (int yw = 0; yw < output_w; ++yw) {
      const T* src = img_data + (yh * stride_h * W + yw * stride_w) * C;
      CopyMatrix<T, CPUContext>(
          kernel_h, kernel_w * C, src, W * C, col_data, kernel_w * C, context);
      col_data += kernel_size * C;
    }
  }
}

template <typename T>
C10_EXPORT void Col2ImZeroPaddingAndNoDilationNHWC(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const T* col_data,
    T* img_data,
    CPUContext* context) {
  Set<T, CPUContext>(H * W * C, T(0), img_data, context);
  const int output_h = (H - kernel_h) / stride_h + 1;
  const int output_w = (W - kernel_w) / stride_w + 1;
  const int kernel_size = kernel_h * kernel_w;
  for (int yh = 0; yh < output_h; ++yh) {
    for (int yw = 0; yw < output_w; ++yw) {
      T* dst = img_data + (yh * stride_h * W + yw * stride_w) * C;
      EigenOuterStridedArrayMap<T>(
          dst, kernel_w * C, kernel_h, EigenOuterStride(W * C)) +=
          ConstEigenArrayMap<T>(col_data, kernel_w * C, kernel_h);
      col_data += kernel_size * C;
    }
  }
}

template <typename T, bool kCol2Im>
C10_EXPORT void Im2ColNdNCHWImpl(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* X_data,
    float* Y_data) {
  if (kCol2Im) {
    std::memset(Y_data, 0, img_size * sizeof(float));
  }
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  std::vector<FixedDivisor<int>> kernel_shape_div(N);
  for (int i = 0; i < N; ++i) {
    kernel_shape_div[i] = FixedDivisor<int>(kernel_shape[i]);
  }
  std::vector<int> d_offset(N, 0);
  std::vector<int> d_iter(N, 0);
  for (int i = 0; i < outer_size; ++i) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = i;
    for (int d_i = N - 1; d_i >= 0; --d_i) {
      kernel_shape_div[d_i].DivMod(offset, &offset, &d_offset[d_i]);
    }
    for (int j = 0; j < inner_size; ++j) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      const int col_index = i * inner_size + j;
      int img_index = i / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < N; ++d_i) {
        const int d_img = d_iter[d_i] * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= !utils::IsAGeZeroAndALtB(d_img, img_shape[d_i + 1]);
        img_index = img_index * img_shape[d_i + 1] + d_img;
      }
      if (!kCol2Im) {
        Y_data[col_index] = is_padding ? 0 : X_data[img_index];
      } else if (!is_padding) {
        Y_data[img_index] += X_data[col_index];
      }
      utils::IncreaseIndexInDims(N, col_shape + 1, d_iter.data());
    }
  }
}

template <typename T>
void Im2Col3dNCHWImpl(
    const int channels,
    const int clip_len,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int pad_p,
    const int pad_t,
    const int pad_l,
    const int pad_a,
    const int pad_b,
    const int pad_r,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const T* img_data,
    T* col_data) {
  const int output_t =
      (clip_len + pad_p + pad_a - (dilation_t * (kernel_t - 1) + 1)) /
          stride_t +
      1;
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;
  const int kernel_size = kernel_t * kernel_h * kernel_w;
  const int kernel_hw_size = kernel_h * kernel_w;
  const int output_size = output_t * output_h * output_w;
  const int channel_size = clip_len * height * width;
  const int output_hw_size = output_h * output_w;
  const int channel_hw_size = height * width;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_t == 1 && dilation_h == 1 && dilation_w == 1 && pad_a == 0 &&
      pad_p == 0 && pad_l == 0 && pad_r == 0 && pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_size; k++) {
      const auto nip = k / kernel_size;
      const auto rest = k % kernel_size;
      const auto kt = rest / kernel_hw_size;
      const auto rest_hw = rest % kernel_hw_size;
      const auto kh = rest_hw / kernel_w;
      const auto kw = rest_hw % kernel_w;
      auto* dst = col_data + nip * (kernel_size * output_size) +
          kt * (kernel_hw_size * output_size) + kh * (kernel_w * output_size) +
          kw * output_size;
      const auto* src = img_data + nip * channel_size;
      for (auto t = 0; t < output_t; t++) {
        const auto it = t * stride_t + kt;
        for (auto y = 0; y < output_h; y++) {
          const auto iy = y * stride_h + kh;
          const auto ix = kw;
          if (stride_w == 1) {
            memcpy(
                dst + (t * output_hw_size + y * output_w),
                src + (it * channel_hw_size + iy * width + ix),
                sizeof(T) * output_w);
          } else {
            for (auto x = 0; x < output_w; x++) {
              memcpy(
                  dst + (t * output_hw_size + y * output_w + x),
                  src + (it * channel_hw_size + iy * width + ix + x * stride_w),
                  sizeof(T));
            }
          }
        }
      }
    }
    return;
  }
  // Fast path for equal padding
  if (pad_a == pad_p && pad_l == pad_r && pad_t == pad_b) {
    const int pad_f = pad_a;
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    for (int channel = channels; channel--; img_data += channel_size) {
      for (int kernel_frame = 0; kernel_frame < kernel_t; kernel_frame++) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
          for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_frame = -pad_f + kernel_frame * dilation_t;
            for (int output_frames = output_t; output_frames; output_frames--) {
              if (!utils::IsAGeZeroAndALtB(input_frame, clip_len)) {
                for (int output_rows = output_h; output_rows; output_rows--) {
                  for (int output_cols = output_w; output_cols; output_cols--) {
                    *(col_data++) = 0;
                  }
                }
              } else {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                  if (!utils::IsAGeZeroAndALtB(input_row, height)) {
                    for (int output_cols = output_w; output_cols;
                         output_cols--) {
                      *(col_data++) = 0;
                    }
                  } else {
                    int input_col = -pad_w + kernel_col * dilation_w;
                    for (int output_col = output_w; output_col; output_col--) {
                      if (utils::IsAGeZeroAndALtB(input_col, width)) {
                        *(col_data++) = img_data
                            [(input_frame * height + input_row) * width +
                             input_col];
                      } else {
                        *(col_data++) = 0;
                      }
                      input_col += stride_w;
                    }
                  }
                  input_row += stride_h;
                }
              }
              input_frame += stride_t;
            }
          }
        }
      }
    }
    return;
  }

  // Baseline
  const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int clip_col = (clip_len + pad_p + pad_a - dkernel_t) / stride_t + 1;
  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_t * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int t_offset = (c / kernel_w / kernel_h) % kernel_t;
    int c_im = c / kernel_h / kernel_w / kernel_t;
    for (int t = 0; t < clip_col; ++t) {
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int t_pad = t * stride_t - pad_p + t_offset * dilation_t;
          int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
          int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
          if (t_pad >= 0 && t_pad < clip_len && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width) {
            col_data[((c * clip_col + t) * height_col + h) * width_col + w] =
                img_data
                    [((c_im * clip_len + t_pad) * height + h_pad) * width +
                     w_pad];
          } else {
            col_data[((c * clip_col + t) * height_col + h) * width_col + w] = 0;
          }
        }
      }
    }
  }
}

} // namespace

template <>
C10_EXPORT void Im2ColNd<float, CPUContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    CPUContext* /* context */,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Im2Col.
  if (N == 3) {
    const int channels =
        col_shape[0] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
    Im2Col3dNCHWImpl<float>(
        channels,
        img_shape[1],
        img_shape[2],
        img_shape[3],
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        dilation[0],
        dilation[1],
        dilation[2],
        pad[0],
        pad[1],
        pad[2],
        pad[3],
        pad[4],
        pad[5],
        stride[0],
        stride[1],
        stride[2],
        img_data,
        col_data);
  } else {
    Im2ColNdNCHWImpl<float, false>(
        N,
        img_size,
        col_size,
        img_shape,
        col_shape,
        kernel_shape,
        stride,
        dilation,
        pad,
        img_data,
        col_data);
  }
}

template <>
C10_EXPORT void Col2ImNd<float, CPUContext, StorageOrder::NCHW>(
    const int N,
    const int img_size,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    CPUContext* /* context */,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Col2Im.
  Im2ColNdNCHWImpl<float, true>(
      N,
      img_size,
      col_size,
      img_shape,
      col_shape,
      kernel_shape,
      stride,
      dilation,
      pad,
      col_data,
      img_data);
}

template <>
C10_EXPORT void Im2Col<float, CPUContext, StorageOrder::NCHW>(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    CPUContext* context,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Im2Col.

  // Fast path for zero padding and no dilation
  if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
      dilation_w == 1) {
    Im2ColZeroPaddingAndNoDilationNCHW<float>(
        C,
        H,
        W,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        img_data,
        col_data,
        context);
    return;
  }

  // Baseline
  const int output_h =
      (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int output_size = output_h * output_w;
  for (int c = 0; c < C; ++c) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        for (int h = 0; h < output_h; ++h) {
          const int h_pad = h * stride_h - pad_t + kh * dilation_h;
          if (!utils::IsAGeZeroAndALtB(h_pad, H)) {
            std::memset(col_data + h * output_w, 0, output_w * sizeof(float));
            continue;
          }
          for (int w = 0; w < output_w; ++w) {
            const int w_pad = w * stride_w - pad_l + kw * dilation_w;
            col_data[h * output_w + w] = utils::IsAGeZeroAndALtB(w_pad, W)
                ? img_data[(c * H + h_pad) * W + w_pad]
                : 0;
          }
        }
        col_data += output_size;
      }
    }
  }
}

template <>
C10_EXPORT void Im2Col<float, CPUContext, StorageOrder::NHWC>(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* img_data,
    float* col_data,
    CPUContext* context,
    const int groups) {
  // Fast path for zero padding and no dilation
  if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
      dilation_w == 1 && groups == 1) {
    Im2ColZeroPaddingAndNoDilationNHWC<float>(
        C,
        H,
        W,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        img_data,
        col_data,
        context);
    return;
  }

  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int h_pad = -pad_t;
  if (groups == 1) {
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
          if (!utils::IsAGeZeroAndALtB(ih, H)) {
            std::memset(col_data, 0, sizeof(float) * kernel_w * C);
            col_data += kernel_w * C;
            continue;
          }
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
            if (utils::IsAGeZeroAndALtB(iw, W)) {
              std::memcpy(
                  col_data, img_data + (ih * W + iw) * C, sizeof(float) * C);
            } else {
              std::memset(col_data, 0, sizeof(float) * C);
            }
            col_data += C;
          } // iw
        } // ih
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  } else {
    /**
     * img_data in N H W G C/G layout
     * col_data in N G H W R S C/G layout
     * Note that groups are pulled out to an outer dimension so that we can use
     * GEMMs efficiently.
     */
    const int C_per_G = C / groups;
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        int r = 0;
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
          int s = 0;
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
            if (utils::IsAGeZeroAndALtB(ih, H) &&
                utils::IsAGeZeroAndALtB(iw, W)) {
              for (int g = 0; g < groups; ++g) {
                std::memcpy(
                    col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                    img_data + (ih * W + iw) * C + g * C_per_G,
                    sizeof(float) * C_per_G);
              }
            } else {
              for (int g = 0; g < groups; ++g) {
                std::memset(
                    col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                    0,
                    sizeof(float) * C_per_G);
              }
            }
          } // iw
        } // ih
        col_data += kernel_h * kernel_w * C;
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  }
}

/**
 * The layout of the result is N H W G R S C/G.
 * Note that groups are pulled out to an outer dimension so that we can use
 * GEMMs efficiently.
 */
template <typename TData>
C10_EXPORT void Im2Col3dNHWCImpl(
    const int C,
    const int T,
    const int H,
    const int W,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int pad_p, // previous frame
    const int pad_t, // top
    const int pad_l, // left
    const int pad_n, // next frame
    const int pad_b, // bottom
    const int pad_r, // right
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const TData* img_data,
    TData* col_data,
    const int groups) {
  const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_t = (T + pad_p + pad_n - dkernel_t) / stride_t + 1;
  const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int C_per_G = C / groups;
  int t_pad = -pad_p;
  for (int t = 0; t < output_t; ++t) {
    int h_pad = -pad_t;
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        int q = 0;
        for (int it = t_pad; it < t_pad + dkernel_t; it += dilation_t, ++q) {
          int r = 0;
          for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
            int s = 0;
            for (int iw = w_pad; iw < w_pad + dkernel_w;
                 iw += dilation_w, ++s) {
              if (utils::IsAGeZeroAndALtB(it, T) &&
                  utils::IsAGeZeroAndALtB(ih, H) &&
                  utils::IsAGeZeroAndALtB(iw, W)) {
                for (int g = 0; g < groups; ++g) {
                  std::memcpy(
                      col_data +
                          (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                              C_per_G,
                      img_data + ((it * H + ih) * W + iw) * C + g * C_per_G,
                      sizeof(TData) * C_per_G);
                }
              } else {
                for (int g = 0; g < groups; ++g) {
                  std::memset(
                      col_data +
                          (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                              C_per_G,
                      0,
                      sizeof(TData) * C_per_G);
                }
              }
            } // iw
          } // ih
        } // it
        col_data += kernel_t * kernel_h * kernel_w * C;
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
    t_pad += stride_t;
  } // t
}

template <>
C10_EXPORT void Im2ColNd<float, CPUContext, StorageOrder::NHWC>(
    const int N,
    const int /*img_size*/,
    const int /*col_size*/,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* img_data,
    float* col_data,
    CPUContext* /* context */,
    const int groups) {
  if (N == 3) {
    const int channels =
        col_shape[3] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
    Im2Col3dNHWCImpl<float>(
        channels,
        img_shape[0],
        img_shape[1],
        img_shape[2],
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        dilation[0],
        dilation[1],
        dilation[2],
        pad[0],
        pad[1],
        pad[2],
        pad[3],
        pad[4],
        pad[5],
        stride[0],
        stride[1],
        stride[2],
        img_data,
        col_data,
        groups);
  } else {
    CAFFE_NOT_IMPLEMENTED;
  }
}

template <>
C10_EXPORT void Col2Im<float, CPUContext, StorageOrder::NCHW>(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    CPUContext* context,
    const int /* groups */) {
  // In NCHW, the number of groups doesn't affect Col2Im.

  // Fast path for zero padding and no dilation
  if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
      dilation_w == 1) {
    Col2ImZeroPaddingAndNoDilationNCHW<float>(
        C,
        H,
        W,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        col_data,
        img_data,
        context);
    return;
  }

  // Fallback
  Set<float, CPUContext>(C * H * W, 0.0f, img_data, context);
  const int output_h =
      (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int output_size = output_h * output_w;
  for (int c = 0; c < C; ++c) {
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        for (int h = 0; h < output_h; ++h) {
          const int h_pad = h * stride_h - pad_t + kh * dilation_h;
          if (!utils::IsAGeZeroAndALtB(h_pad, H)) {
            continue;
          }
          for (int w = 0; w < output_w; ++w) {
            const int w_pad = w * stride_w - pad_l + kw * dilation_w;
            if (utils::IsAGeZeroAndALtB(w_pad, W)) {
              img_data[(c * H + h_pad) * W + w_pad] +=
                  col_data[h * output_w + w];
            }
          }
        }
        col_data += output_size;
      }
    }
  }
}

template <>
C10_EXPORT void Col2Im<float, CPUContext, StorageOrder::NHWC>(
    const int C,
    const int H,
    const int W,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const float* col_data,
    float* img_data,
    CPUContext* context,
    const int groups) {
  // Fast path for zero padding and no dilation
  if (pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0 && dilation_h == 1 &&
      dilation_w == 1 && groups == 1) {
    Col2ImZeroPaddingAndNoDilationNHWC<float>(
        C,
        H,
        W,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        col_data,
        img_data,
        context);
    return;
  }

  Set<float, CPUContext>(H * W * C, 0, img_data, context);
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int h_pad = -pad_t;
  if (groups == 1) {
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
          if (!utils::IsAGeZeroAndALtB(ih, H)) {
            col_data += kernel_w * C;
            continue;
          }
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
            if (utils::IsAGeZeroAndALtB(iw, W)) {
              float* img_data_patch = img_data + (ih * W + iw) * C;
              Add<float, CPUContext>(
                  C, img_data_patch, col_data, img_data_patch, context);
            }
            col_data += C;
          } // iw
        } // ih
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  } else {
    const int C_per_G = C / groups;
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        int r = 0;
        for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
          int s = 0;
          for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
            if (utils::IsAGeZeroAndALtB(ih, H) &&
                utils::IsAGeZeroAndALtB(iw, W)) {
              float* img_data_patch = img_data + (ih * W + iw) * C;
              for (int g = 0; g < groups; ++g) {
                Add<float, CPUContext>(
                    C_per_G,
                    img_data_patch + g * C_per_G,
                    col_data + ((g * kernel_h + r) * kernel_w + s) * C_per_G,
                    img_data_patch + g * C_per_G,
                    context);
              }
            }
          } // iw
        } // ih
        col_data += kernel_h * kernel_w * C;
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
  }
}

/**
 * The layout of the result is N H W G R S C/G.
 * Note that groups are pulled out to an outer dimension so that we can use
 * GEMMs efficiently.
 */
template <typename TData>
C10_EXPORT void Col2Im3dNHWCImpl(
    const int C,
    const int T,
    const int H,
    const int W,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int pad_p, // previous frame
    const int pad_t, // top
    const int pad_l, // left
    const int pad_n, // next frame
    const int pad_b, // bottom
    const int pad_r, // right
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const TData* col_data,
    TData* img_data,
    CPUContext* context,
    const int groups) {
  Set<float, CPUContext>(T * H * W * C, 0, img_data, context);
  const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;
  const int output_t = (T + pad_p + pad_n - dkernel_t) / stride_t + 1;
  const int output_h = (H + pad_t + pad_b - dkernel_h) / stride_h + 1;
  const int output_w = (W + pad_l + pad_r - dkernel_w) / stride_w + 1;
  const int C_per_G = C / groups;

  int t_pad = -pad_p;
  for (int t = 0; t < output_t; ++t) {
    int h_pad = -pad_t;
    for (int h = 0; h < output_h; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < output_w; ++w) {
        int q = 0;
        for (int it = t_pad; it < t_pad + dkernel_t; it += dilation_t, ++q) {
          int r = 0;
          for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
            int s = 0;
            for (int iw = w_pad; iw < w_pad + dkernel_w;
                 iw += dilation_w, ++s) {
              if (utils::IsAGeZeroAndALtB(it, T) &&
                  utils::IsAGeZeroAndALtB(ih, H) &&
                  utils::IsAGeZeroAndALtB(iw, W)) {
                float* img_data_patch = img_data + ((it * T + ih) * W + iw) * C;
                for (int g = 0; g < groups; ++g) {
                  Add<float, CPUContext>(
                      C_per_G,
                      img_data_patch + g * C_per_G,
                      col_data +
                          (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                              C_per_G,
                      img_data_patch + g * C_per_G,
                      context);
                }
              }
            } // iw
          } // ih
        } // it
        col_data += kernel_t * kernel_h * kernel_w * C;
        w_pad += stride_w;
      } // w
      h_pad += stride_h;
    } // h
    t_pad += stride_t;
  } // t
}

template <>
C10_EXPORT void Col2ImNd<float, CPUContext, StorageOrder::NHWC>(
    const int N,
    const int /*img_size*/,
    const int /*col_size*/,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const float* col_data,
    float* img_data,
    CPUContext* context,
    const int groups) {
  if (N == 3) {
    const int channels =
        col_shape[3] / kernel_shape[0] / kernel_shape[1] / kernel_shape[2];
    Col2Im3dNHWCImpl<float>(
        channels,
        img_shape[0],
        img_shape[1],
        img_shape[2],
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        dilation[0],
        dilation[1],
        dilation[2],
        pad[0],
        pad[1],
        pad[2],
        pad[3],
        pad[4],
        pad[5],
        stride[0],
        stride[1],
        stride[2],
        col_data,
        img_data,
        context,
        groups);
  } else {
    CAFFE_NOT_IMPLEMENTED;
  }
}

template <>
C10_EXPORT void BiasCHW<float, CPUContext>(
    const float* bias,
    const float* /*bias_multiplier*/,
    const int bias_channels,
    const int image_size,
    float* image,
    CPUContext* /*context*/) {
  // Sum the per-channel bias into every image plane
  for (int c = 0; c < bias_channels; ++c) {
    float b = bias[c];

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    float32x4_t vBias = vdupq_n_f32(b);

    // We give alignment hints for additional speed, so handle the
    // non-vectorizable prologue separately
    constexpr int kVecSizeInFloat = sizeof(float32x4_t) / sizeof(float);

    // FIXME: if input < kVecSizeInFloat, can't vectorize at all

    int prologue = kVecSizeInFloat -
        // remainder in floats
        (((uintptr_t)image) % (sizeof(float32x4_t))) / sizeof(float);

    int i = 0;
    // Prologue loop
    for (; i < prologue; ++i) {
      image[i] += b;
    }

    // The loop is manually unrolled by 8
    constexpr int kUnroll = 8;
    constexpr int kFloatsPerLoop = kUnroll * kVecSizeInFloat;

    int remainder = image_size - prologue;
    int vectorizable = prologue + (remainder / kFloatsPerLoop) * kFloatsPerLoop;

    // Vectorizable body
    for (; i < vectorizable; i += kFloatsPerLoop) {
      // Manually unrolled
      float32x4_t v0 = vld1q_f32_aligned(image + i + 0);
      float32x4_t v1 = vld1q_f32_aligned(image + i + 4);
      float32x4_t v2 = vld1q_f32_aligned(image + i + 8);
      float32x4_t v3 = vld1q_f32_aligned(image + i + 12);
      float32x4_t v4 = vld1q_f32_aligned(image + i + 16);
      float32x4_t v5 = vld1q_f32_aligned(image + i + 20);
      float32x4_t v6 = vld1q_f32_aligned(image + i + 24);
      float32x4_t v7 = vld1q_f32_aligned(image + i + 28);

      v0 = vaddq_f32(v0, vBias);
      v1 = vaddq_f32(v1, vBias);
      v2 = vaddq_f32(v2, vBias);
      v3 = vaddq_f32(v3, vBias);
      v4 = vaddq_f32(v4, vBias);
      v5 = vaddq_f32(v5, vBias);
      v6 = vaddq_f32(v6, vBias);
      v7 = vaddq_f32(v7, vBias);

      vst1q_f32_aligned(image + i + 0, v0);
      vst1q_f32_aligned(image + i + 4, v1);
      vst1q_f32_aligned(image + i + 8, v2);
      vst1q_f32_aligned(image + i + 12, v3);
      vst1q_f32_aligned(image + i + 16, v4);
      vst1q_f32_aligned(image + i + 20, v5);
      vst1q_f32_aligned(image + i + 24, v6);
      vst1q_f32_aligned(image + i + 28, v7);
    }

    // Non-vectorizable epilogue
    for (; i < image_size; ++i) {
      image[i] += b;
    }
#else
    // Non-NEON CPU implementation
    for (int i = 0; i < image_size; ++i) {
      image[i] += b;
    }
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

    image += image_size;
  }
}

#define CAFFE2_SPECIALIZED_COPYVECTOR(T)                            \
  template <>                                                       \
  C10_EXPORT void CopyVector<T, CPUContext>(                        \
      const int N, const T* src, T* dst, CPUContext* /*context*/) { \
    if (src != dst && N > 0) {                                      \
      memcpy(dst, src, sizeof(T) * N);                              \
    }                                                               \
  }
CAFFE2_SPECIALIZED_COPYVECTOR(float)
CAFFE2_SPECIALIZED_COPYVECTOR(int32_t)
#undef CAFFE2_SPECIALIZED_COPYVECTOR

} // namespace math
} // namespace caffe2
