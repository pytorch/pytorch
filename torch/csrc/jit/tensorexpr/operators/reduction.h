#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeSum(
    const std::vector<ArgValue>& inputs,
    const c10::optional<ScalarType>& outputType);
TORCH_API Tensor computeMean(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType);
TORCH_API Tensor computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
