#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

void ConstantFoldONNX(Block* b, std::map<std::string, at::Tensor>& paramDict, int opset_version);

}
} // namespace torch
