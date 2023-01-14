#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch::jit::tensorexpr {

Tensor computeBatchNorm(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

} // namespace torch::jit::tensorexpr
