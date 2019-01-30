#ifndef CAFFE2_UTILS_MATH_ELEMENTWISE_H_
#define CAFFE2_UTILS_MATH_ELEMENTWISE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

template <typename T, class Context>
void Exp(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Log(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sin(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Asin(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cos(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Acos(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Tan(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Atan(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sinh(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cosh(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void SinCos(const int N, const T* X, T* S, T* C, Context* context);
template <typename T, class Context>
void Tanh(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Abs(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sqr(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sqrt(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Rsqrt(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cube(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cbrt(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Neg(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sign(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Not(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Powx(const int N, const T* A, const T b, T* Y, Context* context);
template <typename T, class Context>
void Inv(const int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Erf(const int N, const T* X, T* Y, Context* context);

template <typename T, class Context, StorageOrder kOrder>
CAFFE2_API void AffineChannel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y,
    Context* context);

} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_ELEMENTWISE_H_
