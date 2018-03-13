#include "torch/csrc/jit/passes/inplace_check.h"

namespace torch { namespace jit {

void CheckInplace(Block * block) {
  for (auto node : block->nodes()) {
    if (node->kind() == kPythonOp && node->hasAttribute(kinplace)) {
      if (node->i(kinplace)) {
        throw std::runtime_error(std::string("inplace ") +
                                 static_cast<PythonOp*>(node)->name() +
                                 " not supported in the JIT");
      }
    }
  }
}

void CheckInplace(std::shared_ptr<Graph>& graph) {
  CheckInplace(graph->block());
}

}} // namespace torch::jit
