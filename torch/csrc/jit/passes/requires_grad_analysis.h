#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <memory>

namespace torch {
namespace jit {

struct Graph;
struct ArgumentSpec;

TORCH_API void PropagateRequiresGrad(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
