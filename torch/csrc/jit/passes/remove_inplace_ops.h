#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch {
namespace jit {
// see .cpp for docs
TORCH_API void ImplicitCastForBinaryInplaceOps(Block* block);
} // namespace jit
} // namespace torch
