#include <torch/csrc/jit/passes/inplace_check.h>

#include <c10/util/Exception.h>

namespace torch::jit {

static void CheckInplace(Block* block) {
  for (auto node : block->nodes()) {
    if (node->kind() == prim::PythonOp && node->hasAttribute(attr::inplace)) {
      TORCH_CHECK(
          !node->i(attr::inplace),
          "inplace ",
          static_cast<PythonOp*>(node)->name(),
          " not supported in the JIT");
    }
  }
}

void CheckInplace(std::shared_ptr<Graph>& graph) {
  CheckInplace(graph->block());
}

} // namespace torch::jit
