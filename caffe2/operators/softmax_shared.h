#ifndef CAFFE2_OPERATORS_SOFTMAX_SHARED_H_
#define CAFFE2_OPERATORS_SOFTMAX_SHARED_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

void SoftmaxCPU(
    CPUContext& context,
    const int N,
    const int D,
    const Tensor<CPUContext>& X,
    float* Ydata,
    Tensor<CPUContext>& scale,
    Tensor<CPUContext>& sum_multiplier);
} // namespace caffe2

#endif // #define CAFFE2_OPERATORS_SOFTMAX_SHARED_H_
