#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FoldIfONNX(Block* b, bool dynamic_axes);

} // namespace jit

} // namespace torch
