#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API void liftClosuresAndForks(const std::shared_ptr<Graph>& graph);
TORCH_API void lambdaLiftFork(Node* fork_node);

} // namespace script
} // namespace jit
} // namespace torch
