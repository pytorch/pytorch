#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void Inline(Block* block);

} // namespace jit
} // namespace torch
