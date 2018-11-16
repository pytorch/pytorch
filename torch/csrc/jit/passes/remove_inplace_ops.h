#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {
// see .cpp for docs
TORCH_API void RemoveInplaceOps(std::shared_ptr<Graph> graph);
} // namespace jit
} // namespace torch
