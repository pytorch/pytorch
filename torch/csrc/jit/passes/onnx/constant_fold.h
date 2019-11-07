#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

const int ONNX_OPSET_9 = 9;
const int ONNX_OPSET_10 = 10;
const int ONNX_OPSET_11 = 11;
void ConstantFoldONNX(Block* b, std::map<std::string, at::Tensor>& paramDict, int opset_version);

}
} // namespace torch
