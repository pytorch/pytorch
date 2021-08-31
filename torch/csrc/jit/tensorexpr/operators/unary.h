#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeSign(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
