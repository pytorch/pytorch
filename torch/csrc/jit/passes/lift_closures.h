#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API void liftClosures(const std::shared_ptr<Graph>& graph);

} // namespace script
} // namespace jit
} // namespace torch
