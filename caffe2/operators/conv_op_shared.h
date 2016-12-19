#ifndef CAFFE2_OPERATORS_CONV_OP_SHARED_H_
#define CAFFE2_OPERATORS_CONV_OP_SHARED_H_

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

template <typename Context>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<Context>* buffer)> f);
} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_OP_SHARED_H_
