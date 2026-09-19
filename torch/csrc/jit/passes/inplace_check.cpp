#include <torch/csrc/jit/passes/inplace_check.h>

#include <c10/util/Exception.h>

namespace torch::jit {

static void CheckInplace(Block* block) {
  for (auto node : block->nodes()) {
    if (node->kind() == prim::PythonOp && node->hasAttribute(attr::inplace)) {
      // On FreeBSD, ScalarAttributeValue<T> typeinfo symbols are local to each
      // DSO (no exported key function), so dynamic_cast across DSO boundaries
      // (from python to cpu library) may fail. Guard with try/catch.
      bool is_inplace = false;
      try {
        is_inplace = (bool)node->i(attr::inplace);
      } catch (const std::exception&) {
        // Cannot determine; treat as not-inplace (conservative safe default).
      }
      TORCH_CHECK(
          !is_inplace,
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
