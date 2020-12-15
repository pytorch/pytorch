#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FoldIfONNX(Block* b);
bool FoldValueONNX(Node* n);
bool FoldConditionONNX(Node* n);

} // namespace jit

} // namespace torch
