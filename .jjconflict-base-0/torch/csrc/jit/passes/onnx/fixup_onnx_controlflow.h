#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

std::vector<Value*> FixupONNXControlflowNode(Node* n, int opset_version);
void FixupONNXControlflowNodeOutputs(Node* n);

} // namespace torch::jit
