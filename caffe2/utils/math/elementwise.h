#ifndef CAFFE2_UTILS_MATH_ELEMENTWISE_H_
#define CAFFE2_UTILS_MATH_ELEMENTWISE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

template <typename T, class Context>
CAFFE2_API void Exp(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Log(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Sin(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Asin(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Cos(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Acos(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Tan(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Atan(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Sinh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Cosh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void SinCos(int N, const T* X, T* S, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Tanh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Abs(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Sqr(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Sqrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Rsqrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Cube(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Cbrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Neg(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Sign(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Not(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Powx(int N, const T* A, const T b, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Inv(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void Erf(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
CAFFE2_API void CdfNorm(int N, const T* X, T* Y, Context* context);

template <typename T, class Context>
CAFFE2_API void Set(std::int64_t N, T alpha, T* X, Context* context);

template <typename TAlpha, typename TData, class Context>
CAFFE2_API void
Scale(std::int64_t N, TAlpha alpha, const TData* X, TData* Y, Context* context);

// Different from the Scale function above, if alpha is passed in as a pointer,
// we will assume that it lives on the Context device, for example on GPU.
template <typename TAlpha, typename TData, class Context>
CAFFE2_API void Scale(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y,
    Context* context);

template <typename T, class Context>
CAFFE2_API void Add(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Sub(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Mul(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Div(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
CAFFE2_API void Min(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Max(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
CAFFE2_API void And(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Or(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void Xor(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
CAFFE2_API void
BitwiseAnd(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void
BitwiseOr(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
CAFFE2_API void
BitwiseXor(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
CAFFE2_API void EQ(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
CAFFE2_API void NE(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
CAFFE2_API void LT(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
CAFFE2_API void LE(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
CAFFE2_API void GT(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
CAFFE2_API void GE(int N, const T* A, const T* B, bool* C, Context* context);

template <typename TAlpha, typename TData, class Context>
CAFFE2_API void
Axpy(std::int64_t N, TAlpha alpha, const TData* X, TData* Y, Context* context);

// Different from the Axpy function above, if alpha is passed in
// as a pointer, we will assume that it lives on the Context device,
// for example on GPU.
template <typename TAlpha, typename TData, class Context>
CAFFE2_API void Axpy(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y,
    Context* context);

template <typename TAlpha, typename TData, class Context>
CAFFE2_API void Axpby(
    std::int64_t N,
    TAlpha alpha,
    const TData* X,
    TAlpha beta,
    TData* Y,
    Context* context);

template <typename TAlpha, typename TData, class Context>
CAFFE2_API void Axpby(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    const TAlpha* beta,
    TData* Y,
    Context* context);

} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_ELEMENTWISE_H_
