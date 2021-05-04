#include "caffe2/utils/math/elementwise.h"

#include <algorithm>
#include <functional>

#ifdef CAFFE2_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif // CAFFE2_USE_ACCELERATE

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace math {

////////////////////////////////////////////////////////////////////////////////
// MKL VML alternatives.
// Depending on whether we are using MKL, we will delegate the Caffe2 math
// functions that are VML-related to either the VML call or the Eigen
// implementation. If you are setting the flags (such as AVX) right for your CPU
// architecture, usually Eigen will deliver a throughput as fast as the VML
// functions.
////////////////////////////////////////////////////////////////////////////////
#ifdef CAFFE2_USE_MKL

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Func, MKLFunc, ...)     \
  template <>                                                     \
  C10_EXPORT void Func<T, CPUContext>(                            \
      const int N, const T* X, T* Y, CPUContext* /* context */) { \
    MKLFunc(N, X, Y, ##__VA_ARGS__);                              \
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
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log1p, vsLog1p)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Log1p, vdLog1p)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin, vsSin)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sin, vdSin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Asin, vsAsin)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Asin, vdAsin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos, vsCos)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cos, vdCos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Acos, vsAcos)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Acos, vdAcos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Tan, vsTan)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Tan, vdTan)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Atan, vsAtan)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Atan, vdAtan)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sinh, vsSinh)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sinh, vdSinh)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cosh, vsCosh)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cosh, vdCosh)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Abs, vsAbs)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Abs, vdAbs)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, vsSqr)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqr, vdSqr)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt, vsSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqrt, vdSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Rsqrt, vsInvSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Rsqrt, vdInvSqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cbrt, vsCbrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cbrt, vdCbrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Inv, vsInv)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Inv, vdInv)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Erf, vsErf)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Erf, vdErf)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, CdfNorm, vsCdfNorm)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, CdfNorm, vdCdfNorm)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_SINCOS(T, MKLFunc)                                     \
  template <>                                                           \
  C10_EXPORT void SinCos<T, CPUContext>(                                \
      const int N, const T* X, T* S, T* C, CPUContext* /* context */) { \
    MKLFunc(N, X, S, C);                                                \
  }
DELEGATE_SINCOS(float, vsSinCos)
DELEGATE_SINCOS(double, vdSinCos)
#undef DELEGATE_SINCOS

#define DELEGATE_POWX(T, MKLFunc)                                            \
  template <>                                                                \
  C10_EXPORT void Powx<T, CPUContext>(                                       \
      const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { \
    MKLFunc(N, A, b, Y);                                                     \
  }
DELEGATE_POWX(float, vsPowx)
DELEGATE_POWX(double, vdPowx)
#undef DELEGATE_POWX

#define DELEGATE_SIMPLE_BINARY_FUNCTION(T, Func, MKLFunc)                     \
  template <>                                                                 \
  C10_EXPORT void Func<T, CPUContext>(                                        \
      const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { \
    MKLFunc(N, A, B, C);                                                      \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Add, vsAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Add, vdAdd)
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Sub, vsSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Sub, vdSub)
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Mul, vsMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Mul, vdMul)
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Div, vsDiv)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Div, vdDiv)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION

#define DELEGATE_AXPBY(TAlpha, TData, MKLFunc)                                 \
  template <>                                                                  \
  C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                            \
      const std::int64_t N,                                                    \
      const TAlpha alpha,                                                      \
      const TData* X,                                                          \
      const TAlpha beta,                                                       \
      TData* Y,                                                                \
      CPUContext* /* context */) {                                             \
    MKLFunc(                                                                   \
        N, static_cast<TData>(alpha), X, 1, static_cast<TData>(beta), Y, 1);   \
  }                                                                            \
  template <>                                                                  \
  C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                            \
      const std::int64_t N,                                                    \
      const TAlpha* alpha,                                                     \
      const TData* X,                                                          \
      const TAlpha* beta,                                                      \
      TData* Y,                                                                \
      CPUContext* /* context */) {                                             \
    MKLFunc(                                                                   \
        N, static_cast<TData>(*alpha), X, 1, static_cast<TData>(*beta), Y, 1); \
  }
DELEGATE_AXPBY(float, float, cblas_saxpby)
#undef DELEGATE_AXPBY

#else // CAFFE2_USE_MKL

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Func, EigenFunc)        \
  template <>                                                     \
  C10_EXPORT void Func<T, CPUContext>(                            \
      const int N, const T* X, T* Y, CPUContext* /* context */) { \
    EigenVectorArrayMap<T>(Y, N) =                                \
        ConstEigenVectorArrayMap<T>(X, N).EigenFunc();            \
  }
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Exp, exp)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log, log)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Log, log)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Log1p, log1p)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Log1p, log1p)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sin, sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sin, sin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Asin, asin)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Asin, asin)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cos, cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cos, cos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Acos, acos)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Acos, acos)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Tan, tan)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Tan, tan)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Atan, atan)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Atan, atan)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqr, square)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqr, square)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sqrt, sqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sqrt, sqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Rsqrt, rsqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Rsqrt, rsqrt)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Inv, inverse)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Inv, inverse)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define CAFFE2_SPECIALIZED_SINH(T)                                        \
  template <>                                                             \
  C10_EXPORT void Sinh<T, CPUContext>(                                    \
      const int N, const T* X, T* Y, CPUContext* /* context */) {         \
    ConstEigenVectorArrayMap<T> X_arr(X, N);                              \
    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() - (-X_arr).exp()) / T(2); \
  }
CAFFE2_SPECIALIZED_SINH(float)
CAFFE2_SPECIALIZED_SINH(double)
#undef CAFFE2_SPECIALIZED_SINH

#define CAFFE2_SPECIALIZED_COSH(T)                                        \
  template <>                                                             \
  C10_EXPORT void Cosh<T, CPUContext>(                                    \
      const int N, const T* X, T* Y, CPUContext* /* context */) {         \
    ConstEigenVectorArrayMap<T> X_arr(X, N);                              \
    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() + (-X_arr).exp()) / T(2); \
  }
CAFFE2_SPECIALIZED_COSH(float)
CAFFE2_SPECIALIZED_COSH(double)
#undef CAFFE2_SPECIALIZED_COSH

#define CAFFE2_SPECIALIZED_SINCOS(T)                                        \
  template <>                                                               \
  C10_EXPORT void SinCos<T, CPUContext>(                                    \
      const int N, const T* X, T* S, T* C, CPUContext* /* context */) {     \
    EigenVectorArrayMap<T>(S, N) = ConstEigenVectorArrayMap<T>(X, N).sin(); \
    EigenVectorArrayMap<T>(C, N) = ConstEigenVectorArrayMap<T>(X, N).cos(); \
  }
CAFFE2_SPECIALIZED_SINCOS(float)
CAFFE2_SPECIALIZED_SINCOS(double)
#undef CAFFE2_SPECIALIZED_SINCOS

#define CAFFE2_SPECIALIZED_POWX(T)                                           \
  template <>                                                                \
  C10_EXPORT void Powx<T, CPUContext>(                                       \
      const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { \
    EigenVectorArrayMap<T>(Y, N) = ConstEigenVectorArrayMap<T>(A, N).pow(b); \
  }
CAFFE2_SPECIALIZED_POWX(float)
CAFFE2_SPECIALIZED_POWX(double)
#undef CAFFE2_SPECIALIZED_POWX

#define CAFFE2_SPECIALIZED_CBRT(T)                                  \
  template <>                                                       \
  C10_EXPORT void Cbrt<T, CPUContext>(                              \
      const int N, const T* X, T* Y, CPUContext* /* context */) {   \
    std::transform(X, X + N, Y, [](const T x) { return cbrt(x); }); \
  }
CAFFE2_SPECIALIZED_CBRT(float)
CAFFE2_SPECIALIZED_CBRT(double)
#undef CAFFE2_SPECIALIZED_CBRT

#define CAFFE2_SPECIALIZED_ERF(T)                                  \
  template <>                                                      \
  C10_EXPORT void Erf<T, CPUContext>(                              \
      const int N, const T* X, T* Y, CPUContext* /* context */) {  \
    std::transform(X, X + N, Y, [](const T x) { return erf(x); }); \
  }
CAFFE2_SPECIALIZED_ERF(float)
CAFFE2_SPECIALIZED_ERF(double)
#undef CAFFE2_SPECIALIZED_ERF

#define CAFFE2_SPECIALIZED_CDF_NORM(T)                            \
  template <>                                                     \
  C10_EXPORT void CdfNorm<T, CPUContext>(                         \
      const int N, const T* X, T* Y, CPUContext* /* context */) { \
    std::transform(X, X + N, Y, [](const T x) {                   \
      constexpr T kRsqrt2 = 0.7071067811865475;                   \
      return (T(1) + erf(x * kRsqrt2)) * static_cast<T>(0.5);     \
    });                                                           \
  }
CAFFE2_SPECIALIZED_CDF_NORM(float)
CAFFE2_SPECIALIZED_CDF_NORM(double)
#undef CAFFE2_SPECIALIZED_CDF_NORM

#define DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(T, Func, EigenOp)   \
  template <>                                                                 \
  C10_EXPORT void Func<T, CPUContext>(                                        \
      const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { \
    EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N)               \
        EigenOp ConstEigenVectorArrayMap<T>(B, N);                            \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(float, Add, +)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(double, Add, +)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(float, Sub, -)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(double, Sub, -)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(float, Mul, *)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(double, Mul, *)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(float, Div, /)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(double, Div, /)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR

#define CAFFE2_SPECIALIZED_AXPBY(TAlpha, TData)                             \
  template <>                                                               \
  C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                         \
      const std::int64_t N,                                                 \
      const TAlpha alpha,                                                   \
      const TData* X,                                                       \
      const TAlpha beta,                                                    \
      TData* Y,                                                             \
      CPUContext* /* context */) {                                          \
    EigenVectorArrayMap<TData> Y_arr(Y, N);                                 \
    Y_arr = Y_arr * static_cast<TData>(beta) +                              \
        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  \
  }                                                                         \
  template <>                                                               \
  C10_EXPORT void Axpby<TAlpha, TData, CPUContext>(                         \
      const std::int64_t N,                                                 \
      const TAlpha* alpha,                                                  \
      const TData* X,                                                       \
      const TAlpha* beta,                                                   \
      TData* Y,                                                             \
      CPUContext* /* context */) {                                          \
    EigenVectorArrayMap<TData> Y_arr(Y, N);                                 \
    Y_arr = Y_arr * static_cast<TData>(*beta) +                             \
        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); \
  }
CAFFE2_SPECIALIZED_AXPBY(float, float)
#undef CAFFE2_SPECIALIZED_AXPBY

#endif // CAFFE2_USE_MKL

////////////////////////////////////////////////////////////////////////////////
// BLAS alternatives.
// Depending on whether we have specified an external BLAS library or not, we
// will delegate the Caffe math functions that are BLAS-related to either the
// CBLAS call or the Eigen implementation.
////////////////////////////////////////////////////////////////////////////////
#ifdef CAFFE2_USE_EIGEN_FOR_BLAS

#define CAFFE2_SPECIALIZED_SCALE(TAlpha, TData)                               \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha alpha,                                                     \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (X == Y) {                                                             \
      EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(alpha);          \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  \
    }                                                                         \
  }                                                                           \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha* alpha,                                                    \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (X == Y) {                                                             \
      EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(*alpha);         \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_SCALE(float, float)
CAFFE2_SPECIALIZED_SCALE(double, double)
CAFFE2_SPECIALIZED_SCALE(float, double)
#undef CAFFE2_SPECIALIZED_SCALE

#define CAFFE2_SPECIALIZED_AXPY(TAlpha, TData)                              \
  template <>                                                               \
  C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(                          \
      const std::int64_t N,                                                 \
      const TAlpha alpha,                                                   \
      const TData* X,                                                       \
      TData* Y,                                                             \
      CPUContext* /* context */) {                                          \
    EigenVectorArrayMap<TData>(Y, N) +=                                     \
        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  \
  }                                                                         \
  template <>                                                               \
  C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(                          \
      const std::int64_t N,                                                 \
      const TAlpha* alpha,                                                  \
      const TData* X,                                                       \
      TData* Y,                                                             \
      CPUContext* /* context */) {                                          \
    EigenVectorArrayMap<TData>(Y, N) +=                                     \
        ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); \
  }
CAFFE2_SPECIALIZED_AXPY(float, float)
#undef CAFFE2_SPECIALIZED_AXPY

#else // CAFFE2_USE_EIGEN_FOR_BLAS

#ifdef CAFFE2_USE_MKL

#define DELEGATE_SCALE(TAlpha, TData, MKLFunc1, MKLFunc2)            \
  template <>                                                        \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                  \
      const std::int64_t N,                                          \
      const TAlpha alpha,                                            \
      const TData* X,                                                \
      TData* Y,                                                      \
      CPUContext* /* context */) {                                   \
    const int max_int = std::numeric_limits<int32_t>::max();         \
    int batch = N / max_int;                                         \
    int remainder = N % max_int;                                     \
    std::int64_t offset = 0;                                         \
    for (int i = 0; i < batch; i ++) {                               \
      if (Y == X) {                                                  \
        MKLFunc1(max_int, static_cast<TData>(alpha), Y + offset, 1); \
      } else {                                                       \
        MKLFunc2(max_int, static_cast<TData>(alpha), X + offset, 1, TData(0), Y + offset, 1);  \
      }                                                              \
      offset += max_int;                                             \
    }                                                                \
    if (remainder != 0) {                                            \
      if (Y == X) {                                                  \
        MKLFunc1(remainder, static_cast<TData>(alpha), Y + offset, 1); \
      } else {                                                       \
        MKLFunc2(remainder, static_cast<TData>(alpha), X + offset, 1, TData(0), Y + offset, 1);  \
      }                                                              \
    }                                                                \
  }                                                                  \
  template <>                                                        \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                  \
      const std::int64_t N,                                          \
      const TAlpha* alpha,                                           \
      const TData* X,                                                \
      TData* Y,                                                      \
      CPUContext* /* context */) {                                   \
    const int max_int = std::numeric_limits<int32_t>::max();         \
    int batch = N / max_int;                                         \
    int remainder = N % max_int;                                     \
    std::int64_t offset = 0;                                         \
    for (int i = 0; i < batch; i ++) {                               \
      if (Y == X) {                                                  \
        MKLFunc1(max_int, static_cast<TData>(*alpha), Y + offset, 1); \
      } else {                                                       \
        MKLFunc2(max_int, static_cast<TData>(*alpha), X + offset, 1, TData(0), Y + offset, 1);  \
      }                                                              \
      offset += max_int;                                             \
    }                                                                \
    if (remainder != 0) {                                            \
      if (Y == X) {                                                  \
        MKLFunc1(remainder, static_cast<TData>(*alpha), Y + offset, 1); \
      } else {                                                       \
        MKLFunc2(remainder, static_cast<TData>(*alpha), X + offset, 1, TData(0), Y + offset, 1); \
      }                                                              \
    }                                                                \
  }
DELEGATE_SCALE(float, float, cblas_sscal, cblas_saxpby)
DELEGATE_SCALE(double, double, cblas_dscal, cblas_daxpby)
DELEGATE_SCALE(float, double, cblas_dscal, cblas_daxpby)
#undef DELEGATE_SCALE

#else // CAFFE2_USE_MKL

#define DELEGATE_SCALE(TAlpha, TData, BLASFunc)                               \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha alpha,                                                     \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (Y == X) {                                                             \
      BLASFunc(N, static_cast<TData>(alpha), Y, 1);                           \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  \
    }                                                                         \
  }                                                                           \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha* alpha,                                                    \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (Y == X) {                                                             \
      BLASFunc(N, static_cast<TData>(*alpha), Y, 1);                          \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); \
    }                                                                         \
  }
DELEGATE_SCALE(float, float, cblas_sscal)
DELEGATE_SCALE(double, double, cblas_dscal)
DELEGATE_SCALE(float, double, cblas_dscal)
#undef DELEGATE_SCALE

#endif // CAFFE2_USE_MKL

#define DELEGATE_AXPY(TAlpha, TData, BLASFunc)           \
  template <>                                            \
  C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(       \
      const std::int64_t N,                              \
      const TAlpha alpha,                                \
      const TData* X,                                    \
      TData* Y,                                          \
      CPUContext* /* context */) {                       \
    BLASFunc(N, static_cast<TData>(alpha), X, 1, Y, 1);  \
  }                                                      \
  template <>                                            \
  C10_EXPORT void Axpy<TAlpha, TData, CPUContext>(       \
      const std::int64_t N,                              \
      const TAlpha* alpha,                               \
      const TData* X,                                    \
      TData* Y,                                          \
      CPUContext* /* context */) {                       \
    BLASFunc(N, static_cast<TData>(*alpha), X, 1, Y, 1); \
  }
DELEGATE_AXPY(float, float, cblas_saxpy)
#undef DELEGATE_AXPY

#endif // CAFFE2_USE_EIGEN_FOR_BLAS

////////////////////////////////////////////////////////////////////////////////
// Common math functions being used in Caffe that do not have a BLAS or MKL
// equivalent. For all these functions, we will simply implement them either via
// Eigen or via custom code.
////////////////////////////////////////////////////////////////////////////////

#define CAFFE2_SPECIALIZED_SET(T)                                             \
  template <>                                                                 \
  C10_EXPORT void Set<T, CPUContext>(                                         \
      const std::int64_t N, const T alpha, T* Y, CPUContext* /* context */) { \
    if (N == 0) {                                                             \
      return;                                                                 \
    }                                                                         \
    if (alpha == T(0)) {                                                      \
      std::memset(Y, 0, N * sizeof(T));                                       \
    } else {                                                                  \
      EigenVectorArrayMap<T>(Y, N).setConstant(alpha);                        \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_SET(float)
CAFFE2_SPECIALIZED_SET(double)
CAFFE2_SPECIALIZED_SET(int)
CAFFE2_SPECIALIZED_SET(std::int8_t)
CAFFE2_SPECIALIZED_SET(std::int16_t)
CAFFE2_SPECIALIZED_SET(std::int64_t)
CAFFE2_SPECIALIZED_SET(bool)
CAFFE2_SPECIALIZED_SET(char)
CAFFE2_SPECIALIZED_SET(std::uint8_t)
CAFFE2_SPECIALIZED_SET(std::uint16_t)
#undef CAFFE2_SPECIALIZED_SET

#define DELEGATE_SIMPLE_UNARY_FUNCTION(T, Func, EigenFunc)        \
  template <>                                                     \
  C10_EXPORT void Func<T, CPUContext>(                            \
      const int N, const T* X, T* Y, CPUContext* /* context */) { \
    EigenVectorArrayMap<T>(Y, N) =                                \
        ConstEigenVectorArrayMap<T>(X, N).EigenFunc();            \
  }
// Eigen's Tanh implementation is faster than MKL, so use Eigen here.
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Tanh, tanh)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Tanh, tanh)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cube, cube)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define CAFFE2_SPECIALIZED_NEG(T)                                      \
  template <>                                                          \
  C10_EXPORT void Neg<T, CPUContext>(                                  \
      const int N, const T* X, T* Y, CPUContext* /* context */) {      \
    EigenVectorArrayMap<T>(Y, N) = -ConstEigenVectorArrayMap<T>(X, N); \
  }
CAFFE2_SPECIALIZED_NEG(std::int32_t)
CAFFE2_SPECIALIZED_NEG(std::int64_t)
CAFFE2_SPECIALIZED_NEG(float)
CAFFE2_SPECIALIZED_NEG(double)
#undef CAFFE2_SPECIALIZED_NEG

#define CAFFE2_SPECIALIZED_SCALE(TAlpha, TData)                               \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha alpha,                                                     \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (X == Y) {                                                             \
      EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(alpha);          \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(alpha);  \
    }                                                                         \
  }                                                                           \
  template <>                                                                 \
  C10_EXPORT void Scale<TAlpha, TData, CPUContext>(                           \
      const std::int64_t N,                                                   \
      const TAlpha* alpha,                                                    \
      const TData* X,                                                         \
      TData* Y,                                                               \
      CPUContext* /* context */) {                                            \
    if (X == Y) {                                                             \
      EigenVectorArrayMap<TData>(Y, N) *= static_cast<TData>(*alpha);         \
    } else {                                                                  \
      EigenVectorArrayMap<TData>(Y, N) =                                      \
          ConstEigenVectorArrayMap<TData>(X, N) * static_cast<TData>(*alpha); \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_SCALE(std::int32_t, std::int32_t)
CAFFE2_SPECIALIZED_SCALE(std::int64_t, std::int64_t)
#undef CAFFE2_SPECIALIZED_SCALE

#define DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(T, Func, EigenOp)   \
  template <>                                                                 \
  C10_EXPORT void Func<T, CPUContext>(                                        \
      const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { \
    EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N)               \
        EigenOp ConstEigenVectorArrayMap<T>(B, N);                            \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, Add, +)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, Add, +)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, Sub, -)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, Sub, -)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, Mul, *)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, Mul, *)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, Div, /)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, Div, /)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_OPERATOR

#define DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(T, Func, EigenFunc) \
  template <>                                                                 \
  C10_EXPORT void Func<T, CPUContext>(                                        \
      const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { \
    EigenVectorMap<T>(C, N) = ConstEigenVectorArrayMap<T>(A, N).EigenFunc(    \
        ConstEigenVectorArrayMap<T>(B, N));                                   \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(std::int32_t, Min, min)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(std::int64_t, Min, min)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(float, Min, min)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(double, Min, min)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(std::int32_t, Max, max)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(std::int64_t, Max, max)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(float, Max, max)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION(double, Max, max)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION_BY_EIGEN_FUNCTION

#define DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(T, Func, StdFunc)     \
  template <>                                                                 \
  C10_EXPORT void Func<T, CPUContext>(                                        \
      const int N, const T* A, const T* B, T* C, CPUContext* /* context */) { \
    std::transform(A, A + N, B, C, StdFunc);                                  \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    bool,
    And,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::logical_and<bool>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    bool,
    Or,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::logical_or<bool>())
// NOLINTNEXTLINE(modernize-use-transparent-functors)
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(bool, Xor, std::bit_xor<bool>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    bool,
    BitwiseAnd,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_and<bool>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int32_t,
    BitwiseAnd,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_and<std::int32_t>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int64_t,
    BitwiseAnd,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_and<std::int64_t>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    bool,
    BitwiseOr,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_or<bool>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int32_t,
    BitwiseOr,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_or<std::int32_t>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int64_t,
    BitwiseOr,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_or<std::int64_t>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    bool,
    BitwiseXor,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_xor<bool>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int32_t,
    BitwiseXor,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_xor<std::int32_t>())
DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION(
    std::int64_t,
    BitwiseXor,
    // NOLINTNEXTLINE(modernize-use-transparent-functors)
    std::bit_xor<std::int64_t>())
#undef DELEGATE_SIMPLE_BINARY_FUNCTION_BY_STD_FUNCTION

#define DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(T, Func, EigenOp) \
  template <>                                                                \
  C10_EXPORT void Func<T, CPUContext>(                                       \
      const int N,                                                           \
      const T* A,                                                            \
      const T* B,                                                            \
      bool* C,                                                               \
      CPUContext* /* context */) {                                           \
    EigenVectorArrayMap<bool>(C, N) = ConstEigenVectorArrayMap<T>(A, N)      \
        EigenOp ConstEigenVectorArrayMap<T>(B, N);                           \
  }
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, EQ, ==)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, EQ, ==)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, EQ, ==)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, EQ, ==)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, EQ, ==)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, NE, !=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, NE, !=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, NE, !=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, NE, !=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, NE, !=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, LT, <)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, LT, <)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, LT, <)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, LT, <)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, LT, <)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, LE, <=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, LE, <=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, LE, <=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, LE, <=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, LE, <=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, GT, >)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, GT, >)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, GT, >)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, GT, >)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, GT, >)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(bool, GE, >=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int32_t, GE, >=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(std::int64_t, GE, >=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(float, GE, >=)
DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR(double, GE, >=)
#undef DELEGATE_SIMPLE_COMPARE_FUNCTION_BY_EIGEN_OPERATOR

} // namespace math
} // namespace caffe2
