#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

const int ONNX_OPSET_9 = 9;
const int ONNX_OPSET_10 = 10;
const int ONNX_OPSET_11 = 11;
const int ONNX_OPSET_12 = 12;
const int ONNX_OPSET_13 = 13;

namespace onnx_constant_fold {
c10::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues,
    int opset_version);
}

void ConstantFoldONNX(
    Block* b,
    std::map<std::string, IValue>& paramDict,
    int opset_version);

} // namespace jit

} // namespace torch
