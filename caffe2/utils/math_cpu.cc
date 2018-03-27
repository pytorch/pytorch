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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "caffe2/utils/math.h"
#include "caffe2/utils/cpu_neon.h"
#include "caffe2/core/context.h"
#include "Eigen/Core"
#include "Eigen/Dense"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif  // CAFFE2_USE_MKL

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
void Gemm<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
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
  switch (TransA) {
  case CblasNoTrans: {
    switch (TransB) {
    case CblasNoTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, N, K) *
          ConstEigenMatrixMap<float>(A, K, M));
      return;
    case CblasTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, K, N).transpose() *
          ConstEigenMatrixMap<float>(A, K, M));
      return;
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
    }
  }
  case CblasTrans: {
    switch (TransB) {
    case CblasNoTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, N, K) *
          ConstEigenMatrixMap<float>(A, M, K).transpose());
      return;
    case CblasTrans:
      C_mat.noalias() += alpha * (
          ConstEigenMatrixMap<float>(B, K, N).transpose() *
          ConstEigenMatrixMap<float>(A, M, K).transpose());
      return;
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
    }
  }
  default:
    LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransA";
  }
}

template <>
void GemmEx<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
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
  using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
  using StridedMap = Eigen::Map<Eigen::MatrixXf, 0, OuterStride>;
  using ConstStridedMap = Eigen::Map<const Eigen::MatrixXf, 0, OuterStride>;
  auto C_mat = StridedMap(C, N, M, OuterStride(ldc));
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() +=
              alpha * (ConstStridedMap(B, N, K, OuterStride(ldb)) *
                       ConstStridedMap(A, K, M, OuterStride(lda)));
          return;
        case CblasTrans:
          C_mat.noalias() +=
              alpha * (ConstStridedMap(B, K, N, OuterStride(ldb)).transpose() *
                       ConstStridedMap(A, K, M, OuterStride(lda)));
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
      }
    }
    case CblasTrans: {
      switch (TransB) {
        case CblasNoTrans:
          C_mat.noalias() +=
              alpha * (ConstStridedMap(B, N, K, OuterStride(ldb)) *
                       ConstStridedMap(A, M, K, OuterStride(lda)).transpose());
          return;
        case CblasTrans:
          C_mat.noalias() +=
              alpha * (ConstStridedMap(B, K, N, OuterStride(ldb)).transpose() *
                       ConstStridedMap(A, M, K, OuterStride(lda)).transpose());
          return;
        default:
          LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransB";
      }
    }
    default:
      LOG(FATAL) << "Unexpected CBLAS_TRANSPOSE for TransA";
  }
}

template <>
void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* context,
    TensorProto::DataType math_type) {
  EigenVectorMap<float> y_vec(y, TransA == CblasNoTrans ? M : N);
  if (beta == 0) {
    // In Caffe2 we often do a lazy initialization, which may contain NaNs in
    // the float values. As a result, if beta is 0, we explicitly do a setzero.
    y_vec.setZero();
  } else {
    y_vec *= beta;
  }
  switch (TransA) {
    case CblasNoTrans: {
      y_vec.noalias() += alpha * (
          ConstEigenMatrixMap<float>(A, N, M).transpose() *
          ConstEigenVectorMap<float>(x, N));
      return;
    }
    case CblasTrans: {
      y_vec.noalias() += alpha * (
          ConstEigenMatrixMap<float>(A, N, M) *
          ConstEigenVectorMap<float>(x, M));
      return;
    }
    default:
      LOG(FATAL) << "Gemv float found an unexpected CBLAS_TRANSPOSE input.";
  }
}

#define CAFFE2_SPECIALIZED_SCALE(T)                                            \
  template <>                                                                  \
  void Scale<T, CPUContext>(                                                   \
      const int n, const float alpha, const T* x, T* y, CPUContext* context) { \
    EigenVectorMap<T>(y, n) = ConstEigenVectorMap<T>(x, n) * alpha;            \
  }                                                                            \
  template <>                                                                  \
  void Scale<T, CPUContext>(                                                   \
      const int n,                                                             \
      const float* alpha,                                                      \
      const T* x,                                                              \
      T* y,                                                                    \
      CPUContext* context) {                                                   \
    EigenVectorMap<T>(y, n) = ConstEigenVectorMap<T>(x, n) * (*alpha);         \
  }
CAFFE2_SPECIALIZED_SCALE(float)
#undef CAFFE2_SPECIALIZED_SCALE

#define CAFFE2_SPECIALIZED_DOT(T)                                              \
template<>                                                                     \
void Dot<T, CPUContext>(                                                       \
    const int N, const T* a, const T* b, T* y,                                 \
    CPUContext* context) {                                                     \
  *y = ConstEigenVectorMap<T>(a, N).dot(ConstEigenVectorMap<T>(b, N));         \
}
CAFFE2_SPECIALIZED_DOT(float)
#undef CAFFE2_SPECIALIZED_DOT

#define CAFFE2_SPECIALIZED_AXPY(T)                                          \
  template <>                                                               \
  void Axpy<T, CPUContext>(                                                 \
      const int N, const T alpha, const T* x, T* Y, CPUContext* context) {  \
    EigenVectorMap<T>(Y, N) += ConstEigenVectorMap<T>(x, N) * alpha;        \
  }                                                                         \
  template <>                                                               \
  void Axpy<T, CPUContext>(                                                 \
      const int N, const T* alpha, const T* x, T* Y, CPUContext* context) { \
    EigenVectorMap<T>(Y, N) += ConstEigenVectorMap<T>(x, N) * (*alpha);     \
  }
CAFFE2_SPECIALIZED_AXPY(float)
#undef CAFFE2_SPECIALIZED_AXPY

#define CAFFE2_SPECIALIZED_AXPBY(T)                                            \
template <>                                                                    \
void Axpby<T, CPUContext>(const int N, const T alpha, const T* x,              \
                          const T beta, T* y, CPUContext* context) {           \
  EigenVectorMap<T> y_vec(y, N);                                               \
  y_vec = y_vec * beta + ConstEigenVectorMap<T>(x, N) * alpha;                 \
}
CAFFE2_SPECIALIZED_AXPBY(float)
#undef CAFFE2_SPECIALIZED_AXPBY

#else  // CAFFE2_USE_EIGEN_FOR_BLAS

template <>
void Gemm<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
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
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void GemmEx<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
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
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void Gemv<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const int M,
    const int N,
    const float alpha,
    const float* A,
    const float* x,
    const float beta,
    float* y,
    CPUContext* /*context*/,
    TensorProto::DataType /*math_type*/) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

#define CAFFE2_SPECIALIZED_SCALE(T, prefix)                             \
  template <>                                                           \
  void Scale<T, CPUContext>(                                            \
      const int n, const float alpha, const T* x, T* y, CPUContext*) {  \
    if (y != x)                                                         \
      cblas_##prefix##copy(n, x, 1, y, 1);                              \
    cblas_##prefix##scal(n, static_cast<float>(alpha), y, 1);           \
  }                                                                     \
  template <>                                                           \
  void Scale<T, CPUContext>(                                            \
      const int n, const float* alpha, const T* x, T* y, CPUContext*) { \
    if (y != x)                                                         \
      cblas_##prefix##copy(n, x, 1, y, 1);                              \
    cblas_##prefix##scal(n, static_cast<float>(*alpha), y, 1);          \
  }
CAFFE2_SPECIALIZED_SCALE(float, s)
#undef CAFFE2_SPECIALIZED_SCALE

#define CAFFE2_SPECIALIZED_DOT(T, prefix)                       \
  template <>                                                   \
  void Dot<T, CPUContext>(                                      \
      const int N, const T* a, const T* b, T* y, CPUContext*) { \
    *y = cblas_##prefix##dot(N, a, 1, b, 1);                    \
  }
CAFFE2_SPECIALIZED_DOT(float, s)
#undef CAFFE2_SPECIALIZED_DOT

#define CAFFE2_SPECIALIZED_AXPY(T, prefix)                          \
  template <>                                                       \
  void Axpy<T, CPUContext>(                                         \
      const int N, const T alpha, const T* x, T* y, CPUContext*) {  \
    cblas_##prefix##axpy(N, alpha, x, 1, y, 1);                     \
  }                                                                 \
  template <>                                                       \
  void Axpy<T, CPUContext>(                                         \
      const int N, const T* alpha, const T* x, T* y, CPUContext*) { \
    cblas_##prefix##axpy(N, *alpha, x, 1, y, 1);                    \
  }
CAFFE2_SPECIALIZED_AXPY(float, s)
#undef CAFFE2_SPECIALIZED_AXPY

// cblas_[sd]axpby is not a standard blas function, and if MKL is not present,
// we will need to implement it.
#ifdef CAFFE2_USE_MKL
#define CAFFE2_SPECIALIZED_AXPBY(T, prefix)            \
  template <>                                          \
  void Axpby<T, CPUContext>(                           \
      const int N,                                     \
      const T alpha,                                   \
      const T* x,                                      \
      const T beta,                                    \
      T* y,                                            \
      CPUContext*) {                                   \
    cblas_##prefix##axpby(N, alpha, x, 1, beta, y, 1); \
  }
#else  // CAFFE2_USE_MKL
#define CAFFE2_SPECIALIZED_AXPBY(T, prefix)     \
  template <>                                   \
  void Axpby<T, CPUContext>(                    \
      const int N,                              \
      const T alpha,                            \
      const T* x,                               \
      const T beta,                             \
      T* y,                                     \
      CPUContext*) {                            \
    cblas_##prefix##scal(N, beta, y, 1);        \
    cblas_##prefix##axpy(N, alpha, x, 1, y, 1); \
  }
#endif  // CAFFE2_USE_MKL
CAFFE2_SPECIALIZED_AXPBY(float, s)
#undef CAFFE2_SPECIALIZED_AXPBY

#endif  // CAFFE2_USE_EIGEN_FOR_BLAS

template <>
void GemmBatched<float, CPUContext>(
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const int batch_size,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    CPUContext* context,
    Tensor<CPUContext>*, /* scratch */
    TensorProto::DataType /* math_type */) {
  const int a_stride = M * K;
  const int b_stride = K * N;
  const int c_stride = M * N;

#ifdef CAFFE2_USE_MKL
  (void)context;

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  std::vector<const float*> a_array(batch_size, nullptr);
  std::vector<const float*> b_array(batch_size, nullptr);
  std::vector<float*> c_array(batch_size, nullptr);
  for (int i = 0; i < batch_size; ++i) {
    a_array[i] = A + a_stride * i;
    b_array[i] = B + b_stride * i;
    c_array[i] = C + c_stride * i;
  }
  cblas_sgemm_batch(
      CblasRowMajor,
      &TransA,
      &TransB,
      &M,
      &N,
      &K,
      &alpha,
      a_array.data(),
      &lda,
      b_array.data(),
      &ldb,
      &beta,
      c_array.data(),
      &N, // ldc_array
      1,
      &batch_size);
#else // CAFFE2_USE_MKL
  // loop over matrices in the batch
  for (int i = 0; i < batch_size; ++i) {
    math::Gemm<float, CPUContext>(
        TransA,
        TransB,
        M,
        N,
        K,
        alpha,
        A + a_stride * i,
        B + b_stride * i,
        beta,
        C + c_stride * i,
        context);
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// MKL VML alternatives.
// Depending on whether we are using MKL, we will delegate the Caffe math
// functions that are VML-related to either the VML call or the Eigen
// implementation. If you are setting the flags (such as AVX) right for your CPU
// architecture, usually Eigen will deliver a throughput as fast as the VML
// functions.
////////////////////////////////////////////////////////////////////////////////
#ifdef CAFFE2_USE_MKL

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, OriginalFunc, ...)       \
  template <>                                                                \
  void Funcname<T, CPUContext>(const int N, const T* x, T* y, CPUContext*) { \
    OriginalFunc(N, x, y, ##__VA_ARGS__);                                    \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(
    float,
    Exp,
    vmsExp,
    VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)
DELEGATE_SIMPLE_UNARY_FUNCTION(
    double,
    Exp,
    vmdExp,
    VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, vsLn)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Log, vdLn)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos, vsCos)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cos, vdCos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin, vsSin)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sin, vdSin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Abs, vsAbs)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Abs, vdAbs)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt, vsSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqrt, vdSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, InvSqrt, vsInvSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, InvSqrt, vdInvSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, vsSqr)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqr, vdSqr)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_SINCOS_FUNCTION(T, OriginalFunc)           \
  template <>                                               \
  void SinCos<T, CPUContext>(                               \
      const int N, const T* a, T* ys, T* yc, CPUContext*) { \
    OriginalFunc(N, a, ys, yc);                             \
  }
DELEGATE_SINCOS_FUNCTION(float, vsSinCos)
DELEGATE_SINCOS_FUNCTION(double, vdSinCos)
#undef DELEGATE_SINCOS_FUNCTION

#define DELEGATE_POWX_FUNCTION(T, OriginalFunc)                               \
  template <>                                                                 \
  void Powx<T, CPUContext>(const int N, const T* a, T b, T* y, CPUContext*) { \
    OriginalFunc(N, a, b, y);                                                 \
  }
DELEGATE_POWX_FUNCTION(float, vsPowx)
DELEGATE_POWX_FUNCTION(double, vdPowx)
#undef DELEGATE_POWX_FUNCTION

#define DELEGATE_SIMPLE_BINARY_FUNCTION(T, Funcname, OriginalFunc) \
  template <>                                                      \
  void Funcname<T, CPUContext>(                                    \
      const int N, const T* a, const T* b, T* y, CPUContext*) {    \
    OriginalFunc(N, a, b, y);                                      \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Add, vsAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Add, vdAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Sub, vsSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Sub, vdSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Mul, vsMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Mul, vdMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(float,  Div, vsDiv)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Div, vdDiv)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION

#else  // CAFFE2_USE_MKL

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Funcname, expr)                    \
  template <>                                                                \
  void Funcname<T, CPUContext>(const int N, const T* x, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(x, N).array().expr();   \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, log)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos, cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin, sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt, sqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, InvSqrt, rsqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, square)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_SINCOS_FUNCTION(T)                                        \
  template <>                                                              \
  void SinCos<T, CPUContext>(                                              \
      const int N, const T* x, T* ys, T* yc, CPUContext*) {                \
    EigenVectorMap<T>(ys, N) = ConstEigenVectorMap<T>(x, N).array().sin(); \
    EigenVectorMap<T>(yc, N) = ConstEigenVectorMap<T>(x, N).array().cos(); \
  }
DELEGATE_SINCOS_FUNCTION(float)
DELEGATE_SINCOS_FUNCTION(double)
#undef DELEGATE_SINCOS_FUNCTION

#define DELEGATE_POWX_FUNCTION(T)                                             \
  template <>                                                                 \
  void Powx<T, CPUContext>(const int N, const T* a, T b, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, N) = ConstEigenVectorMap<T>(a, N).array().pow(b);    \
  }
DELEGATE_POWX_FUNCTION(float)
#undef DELEGATE_POWX_FUNCTION

#endif  // CAFFE2_USE_MKL


#define EIGEN_SIMPLE_BINARY_FUNCTION(T, Funcname, expr)                        \
template <>                                                                    \
void Funcname<T, CPUContext>(                                                  \
    const int N, const T* a, const T* b, T* y,                                 \
    CPUContext*) {                                                             \
  EigenVectorMap<T>(y, N) =                                                    \
      ConstEigenVectorMap<T>(a, N).array() expr                                \
      ConstEigenVectorMap<T>(b, N).array();                                    \
}

#ifdef CAFFE2_USE_MKL

#define DEFINE_SIMPLE_BINARY_FUNCTION(Funcname, expr)                          \
EIGEN_SIMPLE_BINARY_FUNCTION(int32_t, Funcname, expr)                          \
EIGEN_SIMPLE_BINARY_FUNCTION(int64_t, Funcname, expr)

#else

#define DEFINE_SIMPLE_BINARY_FUNCTION(Funcname, expr)                          \
EIGEN_SIMPLE_BINARY_FUNCTION(float, Funcname, expr)                            \
EIGEN_SIMPLE_BINARY_FUNCTION(int32_t, Funcname, expr)                          \
EIGEN_SIMPLE_BINARY_FUNCTION(int64_t, Funcname, expr)

#endif

DEFINE_SIMPLE_BINARY_FUNCTION(Add, +)
DEFINE_SIMPLE_BINARY_FUNCTION(Sub, -)
DEFINE_SIMPLE_BINARY_FUNCTION(Mul, *)
DEFINE_SIMPLE_BINARY_FUNCTION(Div, /)

#undef EIGEN_SIMPLE_BINARY_FUNCTION
#undef DEFINE_FLOAT_BINARY_FUNCTION


////////////////////////////////////////////////////////////////////////////////
// Common math functions being used in Caffe that do not have a BLAS or MKL
// equivalent. For all these functions, we will simply implement them either via
// Eigen or via custom code.
////////////////////////////////////////////////////////////////////////////////

#define CAFFE2_SPECIALIZED_REDUCEMIN(T)    \
  template <>                              \
  void ReduceMin<T, CPUContext>(           \
      const int N,                         \
      const T* x,                          \
      T* y,                                \
      Tensor<CPUContext>* /*scratch_ptr*/, \
      CPUContext* /*context*/) {           \
    *y = *std::min_element(x, x + N);      \
  }
CAFFE2_SPECIALIZED_REDUCEMIN(float)
#undef CAFFE2_SPECIALIZED_REDUCEMIN

#define CAFFE2_SPECIALIZED_REDUCEMAX(T)    \
  template <>                              \
  void ReduceMax<T, CPUContext>(           \
      const int N,                         \
      const T* x,                          \
      T* y,                                \
      Tensor<CPUContext>* /*scratch_ptr*/, \
      CPUContext* /*context*/) {           \
    *y = *std::max_element(x, x + N);      \
  }
CAFFE2_SPECIALIZED_REDUCEMAX(float)
CAFFE2_SPECIALIZED_REDUCEMAX(int32_t)
CAFFE2_SPECIALIZED_REDUCEMAX(int64_t)

#undef CAFFE2_SPECIALIZED_REDUCEMAX

#define CAFFE2_SPECIALIZED_ROWWISEMAX(T)                         \
  template <>                                                    \
  void RowwiseMax<T, CPUContext>(                                \
      const int N, const int D, const T* x, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, N) =                                    \
        ConstEigenMatrixMap<T>(x, D, N).colwise().maxCoeff();    \
  }
CAFFE2_SPECIALIZED_ROWWISEMAX(float)
#undef CAFFE2_SPECIALIZED_ROWWISEMAX

#define CAFFE2_SPECIALIZED_COLWISEMAX(T)                         \
  template <>                                                    \
  void ColwiseMax<T, CPUContext>(                                \
      const int N, const int D, const T* x, T* y, CPUContext*) { \
    EigenVectorMap<T>(y, D) =                                    \
        ConstEigenMatrixMap<T>(x, D, N).rowwise().maxCoeff();    \
  }
CAFFE2_SPECIALIZED_COLWISEMAX(float)
#undef CAFFE2_SPECIALIZED_COLWISEMAX

#define CAFFE2_SPECIALIZED_ELEMWISEMAX(T)                                   \
  template <>                                                               \
  void ElemwiseMax<T, CPUContext>(                                          \
      const int N, const T* x, const T* y, T* z, CPUContext* /*context*/) { \
    std::transform(x, x + N, y, z, [](const T& x_i, const T& y_i) {         \
      return std::max(x_i, y_i);                                            \
    });                                                                     \
  }
CAFFE2_SPECIALIZED_ELEMWISEMAX(float)
#undef CAFFE2_SPECIALIZED_ELEMWISEMAX

#define CAFFE2_SPECIALIZED_MAXIMUM(T)                                          \
  template <>                                                                  \
  void Maximum<T, CPUContext>(                                                 \
      const int N, const float alpha, const T* x, T* y, CPUContext* context) { \
    std::transform(                                                            \
        x, x + N, y, [&alpha](const T& x_i) { return std::max(x_i, alpha); }); \
  }
CAFFE2_SPECIALIZED_MAXIMUM(float)
#undef CAFFE2_SPECIALIZED_MAXIMUM

// AddToRow and AddToCol adds the corresponding row/col vector b to the matrix a
// of shape M x N. The actual implementation uses eigen which is column major,
// so notice the row/column swap in the actual implementation.
#define DELEGATE_BROADCAST_BINARY_FUNCTION(T, Funcname, expr)                \
  template <>                                                                \
  void Funcname##ToRow<T, CPUContext>(                                       \
      const int M, const int N, const T* a, const T* b, T* y, CPUContext*) { \
    EigenArrayMap<T>(y, N, M) = ConstEigenArrayMap<T>(a, N, M).colwise()     \
                                    expr ConstEigenVectorArrayMap<T>(b, N);  \
  }                                                                          \
  /* inplace versions */                                                     \
  template <>                                                                \
  void Funcname##ToRow<T, CPUContext>(                                       \
      const int M, const int N, const T* x, T* y, CPUContext*) {             \
    EigenArrayMap<T>(y, N, M).colwise() expr## =                             \
        ConstEigenVectorArrayMap<T>(x, N);                                   \
  }                                                                          \
  template <>                                                                \
  void Funcname##ToCol<T, CPUContext>(                                       \
      const int M, const int N, const T* x, T* y, CPUContext*) {             \
    EigenArrayMap<T>(y, N, M).rowwise() expr## =                             \
        ConstEigenVectorArrayMap<T>(x, M).transpose();                       \
  }

#define DEFINE_BROADCAST_BINARY_FUNCTION(name, op)                       \
  DELEGATE_BROADCAST_BINARY_FUNCTION(int32_t, name, op)                  \
  DELEGATE_BROADCAST_BINARY_FUNCTION(int64_t, name, op)                  \
  DELEGATE_BROADCAST_BINARY_FUNCTION(float, name, op)                    \

DEFINE_BROADCAST_BINARY_FUNCTION(Add, +)
DEFINE_BROADCAST_BINARY_FUNCTION(Sub, -)
DEFINE_BROADCAST_BINARY_FUNCTION(Mul, *)
DEFINE_BROADCAST_BINARY_FUNCTION(Div, /)

#undef DEFINE_BROADCAST_BINARY_FUNCTION
#undef DELEGATE_BROADCAST_BINARY_FUNCTION

#define CAFFE2_SPECIALIZED_SET(T)                                             \
  template <>                                                                 \
  void Set<T, CPUContext>(const size_t N, const T alpha, T* Y, CPUContext*) { \
    if (alpha == (T)0) {                                                      \
      if (Y != nullptr) {                                                     \
        memset(Y, 0, N * sizeof(T));                                          \
      }                                                                       \
    } else {                                                                  \
      EigenVectorMap<T>(Y, N).setConstant(alpha);                             \
    }                                                                         \
  }

CAFFE2_SPECIALIZED_SET(float);
CAFFE2_SPECIALIZED_SET(double);
CAFFE2_SPECIALIZED_SET(int8_t);
CAFFE2_SPECIALIZED_SET(int16_t);
CAFFE2_SPECIALIZED_SET(int);
CAFFE2_SPECIALIZED_SET(int64_t);
CAFFE2_SPECIALIZED_SET(bool);
CAFFE2_SPECIALIZED_SET(char);
CAFFE2_SPECIALIZED_SET(uint8_t);
CAFFE2_SPECIALIZED_SET(uint16_t);
#undef CAFFE2_SPECIALIZED_SET

#define CAFFE2_INSTANTIATE_BINARY_OP(name, op, T)                  \
  template <>                                                      \
  void name<T, CPUContext>(                                        \
      const int n, const T* a, const T* b, bool* y, CPUContext*) { \
    for (int i = 0; i < n; ++i) {                                  \
      y[i] = a[i] op b[i];                                         \
    }                                                              \
  }                                                                \
  template <>                                                      \
  void name##ToRow<T, CPUContext>(                                 \
      const int m,                                                 \
      const int n,                                                 \
      const T* a,                                                  \
      const T* b,                                                  \
      bool* y,                                                     \
      CPUContext*) {                                               \
    for (int i = 0; i < n * m; ++i) {                              \
      y[i] = a[i] op b[i % n];                                     \
    }                                                              \
  }

#define CAFFE2_DEFINE_BINARY_OP(name, op)         \
  CAFFE2_INSTANTIATE_BINARY_OP(name, op, float)   \
  CAFFE2_INSTANTIATE_BINARY_OP(name, op, int32_t) \
  CAFFE2_INSTANTIATE_BINARY_OP(name, op, int64_t)

CAFFE2_DEFINE_BINARY_OP(LT, <);
CAFFE2_DEFINE_BINARY_OP(LE, <=);
CAFFE2_DEFINE_BINARY_OP(GT, >);
CAFFE2_DEFINE_BINARY_OP(GE, >=);

CAFFE2_INSTANTIATE_BINARY_OP(Or, |, bool);
CAFFE2_INSTANTIATE_BINARY_OP(And, &, bool);
CAFFE2_INSTANTIATE_BINARY_OP(Xor, ^, bool);

template <>
void Not<bool, CPUContext>(
    const int n,
    const bool* x,
    bool* y,
    CPUContext* /*context*/) {
  for (int i = 0; i < n; ++i) {
    y[i] = !x[i];
  }
}

#undef CAFFE2_DEFINE_BINARY_OP
#undef CAFFE2_INSTANTIATE_BINARY_OP

#define CAFFE2_SPECIALIZED_CPU_ADD_STRIPED_BATCH(T)             \
  template <>                                                   \
  void AddStripedBatch(                                         \
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

template <>
void RandUniform<float, CPUContext>(
    const size_t n,
    const float a,
    const float b,
    float* r,
    CPUContext* context) {
  std::uniform_real_distribution<float> distribution(a, b);
  for (auto i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

template <>
void RandUniform<int, CPUContext>(
    const size_t n,
    const int a,
    const int b,
    int* r,
    CPUContext* context) {
  std::uniform_int_distribution<int> distribution(a, b);
  for (auto i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

#define CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(T)                      \
  template <>                                                          \
  void RandUniformUnique<T, CPUContext>(                               \
      const size_t n,                                                  \
      const T a,                                                       \
      const T b,                                                       \
      T* r,                                                            \
      const size_t m,                                                  \
      const T* avoid,                                                  \
      CPUContext* context) {                                           \
    CAFFE_ENFORCE_LE(                                                  \
        n, b - a - m + 1, "Cannot satisfy the unique requirement");    \
    std::unordered_set<T> avoid_set(n);                                \
    if (m) {                                                           \
      avoid_set.insert(avoid, avoid + m);                              \
      CAFFE_ENFORCE_EQ(m, avoid_set.size(), "Avoid should be unique"); \
    }                                                                  \
    std::uniform_int_distribution<T> distribution(a, b);               \
    T v = 0;                                                           \
    for (size_t i = 0; i < n; ++i) {                                   \
      do {                                                             \
        v = distribution(context->RandGenerator());                    \
      } while (avoid_set.count(v));                                    \
      r[i] = v;                                                        \
      avoid_set.insert(v);                                             \
    }                                                                  \
  }

CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(int32_t);
CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE(int64_t);
#undef CAFFE2_SPECIALIZED_RAND_UNIFORM_UNIQUE

template <>
void RandGaussian<float, CPUContext>(
    const size_t n,
    const float mean,
    const float std,
    float* r,
    CPUContext* context) {
  std::normal_distribution<float> distribution(mean, std);
  for (auto i = 0; i < n; ++i) {
    r[i] = distribution(context->RandGenerator());
  }
}

#define CAFFE2_SPECIALIZED_SUM(T)            \
  template <>                                \
  void Sum<T, CPUContext>(                   \
      const int N,                           \
      const T* x,                            \
      T* y,                                  \
      CPUContext* /* unused */,              \
      Tensor<CPUContext>* /* unused */) {    \
    *y = ConstEigenVectorMap<T>(x, N).sum(); \
  }

CAFFE2_SPECIALIZED_SUM(float);
CAFFE2_SPECIALIZED_SUM(int32_t);
CAFFE2_SPECIALIZED_SUM(int64_t);

#undef CAFFE2_SPECIALIZED_SUM

template <>
void SumSqr<float, CPUContext>(
    const int N,
    const float* x,
    float* y,
    CPUContext* /*context*/ /* unused */,
    Tensor<CPUContext>* /*scratch_ptr*/ /* unused */) {
  *y = ConstEigenVectorMap<float>(x, N).squaredNorm();
}

template <>
void Select<float, CPUContext>(
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
// Ported from caffe 1.
template <>
void Im2colNd<float, CPUContext, StorageOrder::NCHW>(
    const float* data_img,
    const int* im_shape,
    const int* col_shape,
    const int /* img_size*/,
    const int /* col_size*/,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const int N,
    float* data_col,
    CPUContext* /* context */,
    bool accumulate_output) {
  int kernel_size = 1;
  for (int i = 0; i < N; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(N, 0);
  vector<int> d_iter(N, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = N - 1; d_i >= 0; --d_i) {
      if (d_i < N - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented;) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < N; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im =
            d * stride[d_i] - pad[d_i] + d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (!accumulate_output) {
        if (is_padding) {
          data_col[index_col] = 0;
        } else {
          data_col[index_col] = data_img[index_im];
        }
      } else if (!is_padding) { // col2im
        data_col[index_im] += data_img[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = N - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else { // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    } // while(incremented) {
  } // for (int c = 0; c < channels_col; ++c) {
}

template <>
void Col2imNd<float, CPUContext, StorageOrder::NCHW>(
    const float* data_col,
    const int* img_shape,
    const int* col_shape,
    const int img_size,
    const int col_size,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const int N,
    float* data_img,
    CPUContext* context) {
  Set<float, CPUContext>(img_size, 0, data_img, context);
  Im2colNd<float, CPUContext, StorageOrder::NCHW>(
      data_col,
      img_shape,
      col_shape,
      img_size,
      col_size,
      kernel_shape,
      stride,
      dilation,
      pad,
      N,
      data_img,
      context,
      true);
}

template <>
void Im2col<float, CPUContext, StorageOrder::NCHW>(
    const float* data_im,
    const int channels,
    const int height,
    const int width,
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
    float* data_col,
    CPUContext* /*context*/) {
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
          kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      const auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(
              dst + (y * output_w),
              src + (iy * width + ix),
              sizeof(float) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(
                dst + (y * output_w + x),
                src + (iy * width + ix + x * stride_w),
                sizeof(float));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

template <>
void Im2col<float, CPUContext, StorageOrder::NHWC>(
    const float* data_im,
    const int channels,
    const int height,
    const int width,
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
    float* data_col,
    CPUContext* /*context*/) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(data_col, data_im + (ih * width + iw) * channels,
                   sizeof(float) * channels);
          } else {
            // This should be simply padded with zero.
            memset(data_col, 0, sizeof(float) * channels);
          }
          data_col += channels;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

template <>
void Col2im<float, CPUContext, StorageOrder::NCHW>(
    const float* data_col,
    const int channels,
    const int height,
    const int width,
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
    float* data_im,
    CPUContext* context) {
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  Set<float, CPUContext>(height * width * channels, 0, data_im, context);

  // Fast path for zero padding and no dilation
  // From Torch, modified THNN_(unfolded_acc)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      const auto* dst = data_col +
          nip * (kernel_h * kernel_w * output_h * output_w) +
          kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          auto offsrc = src + (iy * width + ix);
          const auto offdst = dst + (y * output_w);
          for (auto i = 0; i < output_w; ++i) {
            offsrc[i] += offdst[i];
          }
        } else {
          for (auto x = 0; x < output_w; x++) {
            auto offsrc = src + (iy * width + ix + x * stride_w);
            const auto offdst = dst + (y * output_w + x);
            *offsrc += *offdst;
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              data_col += output_w;
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  data_im[input_row * width + input_col] += *data_col;
                }
                data_col++;
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Fallback
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
        }
      }
    }
  }
}

template <>
void Col2im<float, CPUContext, StorageOrder::NHWC>(
    const float* data_col,
    const int channels,
    const int height,
    const int width,
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
    float* data_im,
    CPUContext* context) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  Set<float, CPUContext>(height * width * channels, 0, data_im, context);
  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            auto* data_im_patch = data_im + (ih * width + iw) * channels;
            Add<float, CPUContext>(
                  channels, data_im_patch, data_col, data_im_patch, context);
          }
          data_col += channels;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

template <>
void BiasCHW<float, CPUContext>(
    const float* bias,
    const int bias_channels,
    const int image_size,
    float* image,
    CPUContext* /*context*/) {
  // Sum the per-channel bias into every image plane
  for (int c = 0; c < bias_channels; ++c) {
    float b = bias[c];

#ifdef __ARM_NEON__
    float32x4_t vBias = vdupq_n_f32(b);

    // We give alignment hints for additional speed, so handle the
    // non-vectorizable prologue separately
    constexpr int kVecSizeInFloat = sizeof(float32x4_t) / sizeof(float);

    // FIXME: if input < kVecSizeInFloat, can't vectorize at all

    int prologue =
      kVecSizeInFloat -
      // remainder in floats
      (((uintptr_t) image) % (sizeof(float32x4_t))) / sizeof(float);

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
#endif // __ARM_NEON__

    image += image_size;
  }
}

template <>
void CopyMatrix<CPUContext>(
    const size_t itemsize,
    const int M,
    const int N,
    const void* A,
    const int lda,
    void* B,
    const int ldb,
    CPUContext* /*context*/,
    TypeMeta::TypedCopy copy) {
  if (A == nullptr || B == nullptr) {
    return;
  }
  if (lda == N && ldb == N) {
    // can coalese to a single memcpy of size M * N
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

#define CAFFE2_SPECIALIZED_COPYVECTOR(T)                            \
  template <>                                                       \
  void CopyVector<T, CPUContext>(                                   \
      const int N, const T* src, T* dst, CPUContext* /*context*/) { \
    if (src != dst && N > 0) {                                      \
      memcpy(dst, src, sizeof(T) * N);                              \
    }                                                               \
  }
CAFFE2_SPECIALIZED_COPYVECTOR(float)
#undef CAFFE2_SPECIALIZED_COPYVECTOR

namespace {

#ifdef CAFFE2_USE_HPTT

bool TryTransposeWithHPTT(
    const int num_axes,
    const int* dims,
    const int* axes,
    const float* X,
    float* Y) {
  std::vector<int> axes_cm(num_axes);
  std::vector<int> dims_cm(num_axes);

  // Convert row-major index to column-major.
  const auto cm_fn = [num_axes](const int i) { return num_axes - i - 1; };
  for (int i = 0; i < num_axes; ++i) {
    axes_cm[i] = cm_fn(axes[cm_fn(i)]);
    dims_cm[i] = dims[cm_fn(i)];
  }
  auto plan = hptt::create_plan(
      axes_cm.data(),
      num_axes,
      1.0,
      X,
      dims_cm.data(),
      nullptr,
      0.0,
      Y,
      nullptr,
      hptt::ESTIMATE,
      1);
  if (plan == nullptr) {
    return false;
  }
  plan->execute();
  return true;
}

#endif // CAFFE2_USE_HPTT

std::vector<int>
ComputeXStrides(const int num_axes, const int* dims, const int* axes) {
  std::vector<int> x_strides(num_axes);
  std::vector<int> buff(num_axes);
  int cur_stride = 1;
  for (int i = num_axes - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= dims[i];
  }
  for (int i = 0; i < num_axes; ++i) {
    x_strides[i] = buff[axes[i]];
  }
  return x_strides;
}

void IncreaseIndex(const int* dims, std::vector<int>* index) {
  for (int i = index->size() - 1; i >= 0; --i) {
    ++index->at(i);
    if (index->at(i) >= dims[i]) {
      index->at(i) -= dims[i];
    } else {
      break;
    }
  }
}

template <typename T>
void TransposeCPU(
    const int num_axes,
    const int* x_dims,
    const int* y_dims,
    const int* axes,
    const int data_size,
    const T* X,
    T* Y) {
  // Measure amount of contiguous data we can copy at once
  int block_size = 1;
  int num_shared_idxs = 0;
  for (int i = num_axes - 1; i >= 0 && axes[i] == i; --i) {
    block_size *= y_dims[i];
    ++num_shared_idxs;
  }

  if (num_axes < 2 || num_shared_idxs == num_axes) {
    memcpy(Y, X, data_size * sizeof(T));
    return;
  }

  const int itr_axes = num_axes - num_shared_idxs;
  const std::vector<int> x_strides = ComputeXStrides(itr_axes, x_dims, axes);
  std::vector<int> index_digits(itr_axes, 0);
  const int num_blocks = data_size / block_size;
  for (int y_index = 0; y_index < num_blocks; ++y_index) {
    const int x_index = std::inner_product(
        x_strides.cbegin(), x_strides.cend(), index_digits.cbegin(), 0);
    if (block_size == 1) {
      Y[y_index] = X[x_index];
    } else {
      memcpy(
          Y + block_size * y_index,
          X + block_size * x_index,
          block_size * sizeof(T));
    }
    IncreaseIndex(y_dims, &index_digits);
  }
}

} // namespace

template <>
void Transpose<float, CPUContext>(
    const int num_axes,
    const int* x_dims,
    const int* y_dims,
    const int* axes,
    const int data_size,
    const float* X,
    float* Y,
    CPUContext* /* context */) {
#ifdef CAFFE2_USE_HPTT
  if (TryTransposeWithHPTT(num_axes, x_dims, axes, X, Y)) {
    return;
  }
#endif // CAFFE2_USE_HPTT
  TransposeCPU(num_axes, x_dims, y_dims, axes, data_size, X, Y);
}

#define CAFFE2_SPECIALIZED_TRANSPOSE(T)                            \
  template <>                                                      \
  void Transpose<T, CPUContext>(                                   \
      const int num_axes,                                          \
      const int* x_dims,                                           \
      const int* y_dims,                                           \
      const int* axes,                                             \
      const int data_size,                                         \
      const T* X,                                                  \
      T* Y,                                                        \
      CPUContext* /* context */) {                                 \
    TransposeCPU(num_axes, x_dims, y_dims, axes, data_size, X, Y); \
  }
CAFFE2_SPECIALIZED_TRANSPOSE(double)
CAFFE2_SPECIALIZED_TRANSPOSE(int)
CAFFE2_SPECIALIZED_TRANSPOSE(long)
#undef CAFFE2_SPECIALIZED_TRANSPOSE

} // namespace math
} // namespace caffe2
