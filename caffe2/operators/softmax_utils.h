#ifndef CAFFE2_OPERATORS_SOFTMAX_UTILS_H_
#define CAFFE2_OPERATORS_SOFTMAX_UTILS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
namespace softmax_utils {

template <typename T>
void SoftmaxCPU(
    int N,
    int D,
    bool logarithmic,
    const T* X,
    T* Y,
    T* scratch,
    CPUContext* context);

} // namespace softmax_utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_SOFTMAX_UTILS_H_
