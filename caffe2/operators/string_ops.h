#pragma once
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename ScalarFunctor, typename OutputType = std::string>
using StringElementwiseOp = UnaryElementwiseWithArgsOp<
    TensorTypes<std::string>,
    CPUContext,
    ForEach<ScalarFunctor>,
    OutputType>;
}
