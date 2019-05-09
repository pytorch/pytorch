#ifndef CAFFE2_UTILS_MATH_BROADCAST_H_
#define CAFFE2_UTILS_MATH_BROADCAST_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

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

#endif // CAFFE2_UTILS_MATH_BROADCAST_H_
