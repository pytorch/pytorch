#include <torch/csrc/jit/passes/annotate_warns.h>

#include <atomic>

namespace torch {
namespace jit {

void AnnotateWarns(Block* b) {
  static std::atomic<int64_t> idx(0);
  for (Node* n : b->nodes()) {
    for (Block* child_b : n->blocks()) {
      AnnotateWarns(child_b);
    }

    if (n->kind() != aten::warn) {
      continue;
    }

    n->i_(attr::warn_id, idx);
    idx++;
  }
}

void AnnotateWarns(const std::shared_ptr<Graph>& graph) {
  AnnotateWarns(graph->block());
}

} // namespace jit
} // namespace torch
