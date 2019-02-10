#ifndef CAFFE2_UTILS_MATH_REDUCE_H_
#define CAFFE2_UTILS_MATH_REDUCE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

// Computes mean and variance over axes.
template <typename T, class Context>
CAFFE2_API void Moments(
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
