#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FoldIfNodeONNX(Block* b);
bool ConditionValueONNX(Node* n);
bool IsStaticConditionONNX(Node* n);

} // namespace jit

} // namespace torch
