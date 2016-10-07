#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

template <typename Context>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<Context>* buffer)> f);
}
