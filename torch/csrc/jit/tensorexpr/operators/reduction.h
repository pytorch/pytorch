#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor* computeSum(
    const std::vector<ArgValue>& inputs,
    const c10::optional<ScalarType>& outputType);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
