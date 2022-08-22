#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>

namespace torch {
namespace jit {
namespace tensorexpr {

bool mkldnnPrepackedLinearIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight);
Tensor computeMatmul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeAddMM(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
Tensor computeMkldnnPrepackedLinearRun(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
