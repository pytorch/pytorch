#ifndef CAFFE2_UTILS_MATH_REDUCE_H_
#define CAFFE2_UTILS_MATH_REDUCE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {

class Tensor;

namespace math {

template <typename T, class Context>
TORCH_API void
ReduceMin(const int N, const T* X, T* y, Tensor* scratch_ptr, Context* context);

template <typename T, class Context>
TORCH_API void
ReduceMax(const int N, const T* X, T* y, Tensor* scratch_ptr, Context* context);

// In all of the reduce functions, X_dims and Y_dims should have ndim elements.
// Each dimension of Y_dims must match the corresponding dimension of X_dims or
// must be equal to 1. The dimensions equal to 1 indicate the dimensions of X to
// be reduced.

// Y = alpha * ReduceMin(X)
template <typename T, class Context>
TORCH_API void ReduceMin(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Y = alpha * ReduceMax(X)
template <typename T, class Context>
TORCH_API void ReduceMax(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Y = alpha * ReduceSum(X)
template <typename T, class Context>
TORCH_API void ReduceSum(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Y = alpha * ReduceMean(X)
template <typename T, class Context>
TORCH_API void ReduceMean(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Y = alpha * ReduceL1(X)
template <typename T, class Context>
TORCH_API void ReduceL1(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Y = alpha * ReduceL2(X)
template <typename T, class Context>
TORCH_API void ReduceL2(
    const int ndim,
    const int* X_dims,
    const int* Y_dims,
    const T alpha,
    const T* X,
    T* Y,
    Context* context);

// Computes mean and variance over axes.
template <typename T, class Context>
TORCH_API void Moments(
    const int ndims,
    const int* X_dims,
    const int* Y_dims,
    const T* X,
    T* mean,
    T* var,
    Context* context);

} // namespace math

} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_REDUCE_H_
