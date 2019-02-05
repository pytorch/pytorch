#include "caffe2/utils/math/elementwise.h"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "caffe2/core/context.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {
namespace math {

////////////////////////////////////////////////////////////////////////////////
// MKL VML alternatives.
// Depending on whether we are using MKL, we will delegate the Caffe math
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
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_SINCOS_FUNCTION(T, MKLFunc)                            \
  template <>                                                           \
  C10_EXPORT void SinCos<T, CPUContext>(                                \
      const int N, const T* X, T* S, T* C, CPUContext* /* context */) { \
    MKLFunc(N, X, S, C);                                                \
  }
DELEGATE_SINCOS_FUNCTION(float, vsSinCos)
DELEGATE_SINCOS_FUNCTION(double, vdSinCos)
#undef DELEGATE_SINCOS_FUNCTION

#define DELEGATE_POWX_FUNCTION(T, MKLFunc)                                   \
  template <>                                                                \
  C10_EXPORT void Powx<T, CPUContext>(                                       \
      const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { \
    MKLFunc(N, A, b, Y);                                                     \
  }
DELEGATE_POWX_FUNCTION(float, vsPowx)
DELEGATE_POWX_FUNCTION(double, vdPowx)
#undef DELEGATE_POWX_FUNCTION

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

#define DELEGATE_SINH_FUNCTION(T)                                         \
  template <>                                                             \
  C10_EXPORT void Sinh<T, CPUContext>(                                    \
      const int N, const T* X, T* Y, CPUContext* /* context */) {         \
    ConstEigenVectorArrayMap<T> X_arr(X, N);                              \
    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() - (-X_arr).exp()) / T(2); \
  }
DELEGATE_SINH_FUNCTION(float)
DELEGATE_SINH_FUNCTION(double)
#undef DELEGATE_SINH_FUNCTION

#define DELEGATE_COSH_FUNCTION(T)                                         \
  template <>                                                             \
  C10_EXPORT void Cosh<T, CPUContext>(                                    \
      const int N, const T* X, T* Y, CPUContext* /* context */) {         \
    ConstEigenVectorArrayMap<T> X_arr(X, N);                              \
    EigenVectorArrayMap<T>(Y, N) = (X_arr.exp() + (-X_arr).exp()) / T(2); \
  }
DELEGATE_COSH_FUNCTION(float)
DELEGATE_COSH_FUNCTION(double)
#undef DELEGATE_COSH_FUNCTION

#define DELEGATE_SINCOS_FUNCTION(T)                                         \
  template <>                                                               \
  C10_EXPORT void SinCos<T, CPUContext>(                                    \
      const int N, const T* X, T* S, T* C, CPUContext* /* context */) {     \
    EigenVectorArrayMap<T>(S, N) = ConstEigenVectorArrayMap<T>(X, N).sin(); \
    EigenVectorArrayMap<T>(C, N) = ConstEigenVectorArrayMap<T>(X, N).cos(); \
  }
DELEGATE_SINCOS_FUNCTION(float)
DELEGATE_SINCOS_FUNCTION(double)
#undef DELEGATE_SINCOS_FUNCTION

#define DELEGATE_POWX_FUNCTION(T)                                            \
  template <>                                                                \
  C10_EXPORT void Powx<T, CPUContext>(                                       \
      const int N, const T* A, const T b, T* Y, CPUContext* /* context */) { \
    EigenVectorArrayMap<T>(Y, N) = ConstEigenVectorArrayMap<T>(A, N).pow(b); \
  }
DELEGATE_POWX_FUNCTION(float)
DELEGATE_POWX_FUNCTION(double)
#undef DELEGATE_POWX_FUNCTION

#define DELEGATE_CBRT_FUNCTION(T)                                   \
  template <>                                                       \
  C10_EXPORT void Cbrt<T, CPUContext>(                              \
      const int N, const T* X, T* Y, CPUContext* /* context */) {   \
    std::transform(X, X + N, Y, [](const T x) { return cbrt(x); }); \
  }
DELEGATE_CBRT_FUNCTION(float)
DELEGATE_CBRT_FUNCTION(double)
#undef DELEGATE_CBRT_FUNCTION

#define DELEGATE_ERF_FUNCTION(T)                                   \
  template <>                                                      \
  C10_EXPORT void Erf<T, CPUContext>(                              \
      const int N, const T* X, T* Y, CPUContext* /* context */) {  \
    std::transform(X, X + N, Y, [](const T x) { return erf(x); }); \
  }
DELEGATE_ERF_FUNCTION(float)
DELEGATE_ERF_FUNCTION(double)
#undef DELEGATE_ERF_FUNCTION

#endif // CAFFE2_USE_MKL

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
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Sign, sign)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Abs, abs)
DELEGATE_SIMPLE_UNARY_FUNCTION(float, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(double, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int32_t, Cube, cube)
DELEGATE_SIMPLE_UNARY_FUNCTION(std::int64_t, Cube, cube)
#undef DELEGATE_SIMPLE_UNARY_FUNCTION

#define DELEGATE_NEG_FUNCTION(T)                                       \
  template <>                                                          \
  C10_EXPORT void Neg<T, CPUContext>(                                  \
      const int N, const T* X, T* Y, CPUContext* /* context */) {      \
    EigenVectorArrayMap<T>(Y, N) = -ConstEigenVectorArrayMap<T>(X, N); \
  }
DELEGATE_NEG_FUNCTION(float)
DELEGATE_NEG_FUNCTION(double)
DELEGATE_NEG_FUNCTION(std::int32_t)
DELEGATE_NEG_FUNCTION(std::int64_t)
#undef DELEGATE_NEG_FUNCTION

#define CAFFE2_SPECIALIZED_AFFINE_CHANNEL(T)                \
  template <>                                               \
  void AffineChannel<T, CPUContext, StorageOrder::NCHW>(    \
      const int N,                                          \
      const int C,                                          \
      const int HxW,                                        \
      const T* X,                                           \
      const T* scale,                                       \
      const T* bias,                                        \
      T* Y,                                                 \
      CPUContext* /* context */) {                          \
    ConstEigenVectorArrayMap<T> scale_arr(scale, C);        \
    ConstEigenVectorArrayMap<T> bias_arr(bias, C);          \
    const int stride = C * HxW;                             \
    const T* X_ptr = X;                                     \
    T* Y_ptr = Y;                                           \
    for (int i = 0; i < N; ++i) {                           \
      EigenArrayMap<T>(Y_ptr, HxW, C) =                     \
          (ConstEigenArrayMap<T>(X_ptr, HxW, C).rowwise() * \
           scale_arr.transpose())                           \
              .rowwise() +                                  \
          bias_arr.transpose();                             \
      X_ptr += stride;                                      \
      Y_ptr += stride;                                      \
    }                                                       \
  }                                                         \
  template <>                                               \
  void AffineChannel<T, CPUContext, StorageOrder::NHWC>(    \
      const int N,                                          \
      const int C,                                          \
      const int HxW,                                        \
      const T* X,                                           \
      const T* scale,                                       \
      const T* bias,                                        \
      T* Y,                                                 \
      CPUContext* /* context */) {                          \
    EigenArrayMap<T>(Y, C, N * HxW) =                       \
        (ConstEigenArrayMap<T>(X, C, N * HxW).colwise() *   \
         ConstEigenVectorArrayMap<T>(scale, C))             \
            .colwise() +                                    \
        ConstEigenVectorArrayMap<T>(bias, C);               \
  }
CAFFE2_SPECIALIZED_AFFINE_CHANNEL(float)
#undef CAFFE2_SPECIALIZED_AFFINE_CHANNEL

} // namespace math
} // namespace caffe2
