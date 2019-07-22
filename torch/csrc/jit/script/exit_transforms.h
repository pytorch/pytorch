#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void TransformExits(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
