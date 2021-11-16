#include <torch/csrc/jit/passes/insert_guards.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

void removeProfilingNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes(ib);
      }
    }
  }
}

struct GuardInserter {
  GuardInserter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    insertGuards(graph_->block());
    removeProfilingNodes(graph_->block());
  }

 private:
  void insertGuards(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::profile) {
        auto pttp = n->ty(attr::profiled_type)->cast<TensorType>();
        if (pttp) {
          auto guard = graph_->create(prim::Guard, {n->input()}, 1);
          auto go = guard->output();
          go->setType(pttp);
          guard->insertBefore(n);
          n->output()->replaceAllUsesWith(go);
        } else {
          // we didn't go down this path i.e
          // no profiling information is available
          n->output()->replaceAllUsesWith(n->input());
        }
        it.destroyCurrent();
      } else {
        for (Block* ib : n->blocks()) {
          insertGuards(ib);
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};

void InsertGuards(std::shared_ptr<Graph> graph) {
  GuardInserter gi(std::move(graph));
  gi.run();
}

void RemoveProfilingNodes(const std::shared_ptr<Graph>& graph) {
  removeProfilingNodes(graph->block());
}

} // namespace jit
} // namespace torch
