#ifndef CAFFE2_UTILS_MATH_TRANSPOSE_H_
#define CAFFE2_UTILS_MATH_TRANSPOSE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

// Transpose tensor X with dims by axes and write the result to tensor Y.
template <typename TIndex, typename TData, class Context>
CAFFE2_API void Transpose(
    int ndim,
    const TIndex* dims,
    const int* axes,
    const TData* X,
    TData* Y,
    Context* context);

template <typename T, class Context>
CAFFE2_API void
NCHW2NHWC(int N, int C, int HxW, const T* X, T* Y, Context* context);

template <typename T, class Context>
CAFFE2_API void
NHWC2NCHW(int N, int C, int HxW, const T* X, T* Y, Context* context);

} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_TRANSPOSE_H_
