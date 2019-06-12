#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);

} // namespace script
} // namespace jit
} // namespace torch
