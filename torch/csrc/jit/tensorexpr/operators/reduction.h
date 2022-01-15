#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeSum(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
TORCH_API Tensor computeMean(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
TORCH_API Tensor computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeMax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
