#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

TORCH_API void EraseShapeInformation(const std::shared_ptr<Graph>& graph);
TORCH_API void PropagateInputShapes(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
