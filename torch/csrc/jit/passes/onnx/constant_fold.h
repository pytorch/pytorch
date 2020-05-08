#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

const int ONNX_OPSET_9 = 9;
const int ONNX_OPSET_10 = 10;
const int ONNX_OPSET_11 = 11;
const int ONNX_OPSET_12 = 12;
void ConstantFoldONNX(
    Block* b,
    std::map<std::string, IValue>& paramDict,
    int opset_version);

} // namespace jit
} // namespace torch
