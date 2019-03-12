#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void Inline(Block* block, bool recurse = true);

} // namespace jit
} // namespace torch
