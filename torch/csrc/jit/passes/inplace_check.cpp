#include <torch/csrc/jit/passes/inplace_check.h>

namespace torch {
namespace jit {

static void CheckInplace(Block* block) {
  for (auto node : block->nodes()) {
    if (node->kind() == prim::PythonOp && node->hasAttribute(attr::inplace)) {
      if (node->i(attr::inplace)) {
        throw std::runtime_error(
            std::string("inplace ") + static_cast<PythonOp*>(node)->name() +
            " not supported in the JIT");
      }
    }
  }
}

void CheckInplace(std::shared_ptr<Graph>& graph) {
  CheckInplace(graph->block());
}

} // namespace jit
} // namespace torch
