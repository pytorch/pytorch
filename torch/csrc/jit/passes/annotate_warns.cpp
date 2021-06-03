#include <torch/csrc/jit/passes/annotate_warns.h>

#include <atomic>

namespace torch {
namespace jit {

void annotateWarns(Block* b) {
  static std::atomic<int64_t> idx(0);
  for (Node* n : b->nodes()) {
    for (Block* child_b : n->blocks()) {
      annotateWarns(child_b);
    }

    if (n->kind() != aten::warn) {
      continue;
    }

    n->i_(attr::warn_id, idx);
    idx++;
  }
}

void annotateWarns(const std::shared_ptr<Graph>& graph) {
  annotateWarns(graph->block());
}

} // namespace jit
} // namespace torch
