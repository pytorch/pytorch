#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

TORCH_API void AnnotateEffects(std::shared_ptr<Graph>& graph);
TORCH_API void UnAnnotateEffects(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
