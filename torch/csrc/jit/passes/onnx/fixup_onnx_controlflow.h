#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

std::vector<Value*> FixupONNXControlflowNode(Node* n, int opset_version);

} // namespace jit
} // namespace torch
