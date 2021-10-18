#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API ExprHandle quantizePerTensorQParamFromArg(ArgValue arg);

TORCH_API double immQScale(const BufHandle& qx);

TORCH_API int64_t immQZero(const BufHandle& qx);

TORCH_API ScalarType immQDType(const BufHandle& qx);

TORCH_API bool isQuantized(const BufHandle& qx);

TORCH_API Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

TORCH_API Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);
} // namespace tensorexpr
} // namespace jit
} // namespace torch
