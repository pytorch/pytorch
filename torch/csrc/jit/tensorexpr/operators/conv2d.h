#pragma once

#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch::jit::tensorexpr {

using ScheduleFn = std::function<void(LoopNest&, Tensor*)>;

struct TensorSchedule {
  Tensor* tensor;
  ScheduleFn schedule;
};

TensorSchedule conv2d_depthwise(
    BufHandle input,
    BufHandle weight,
    BufHandle bias,
    int stride,
    int pad,
    int groups);

} // namespace torch::jit::tensorexpr
