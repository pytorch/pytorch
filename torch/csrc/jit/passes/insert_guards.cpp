#include <torch/csrc/jit/passes/insert_guards.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct GuardInserter {
  GuardInserter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    insertGuards(graph_->block());
    removeProfilingNodes(graph_->block());
  }

 private:
  void removeProfilingNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      if (it->kind() == prim::profile) {
        it.destroyCurrent();
      } else {
        for (Block* ib : it->blocks()) {
          removeProfilingNodes(ib);
        }
      }
    }
  }

  void insertGuards(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::profile && n->outputs().size() == 1) {
        auto pttp = n->output()->type()->cast<ProfiledTensorType>();
        if (pttp) {
          // make a *copy* of ProfilingTensorType, in case we'd like
          // to make changes to it independently from the one being
          // profiled
          auto guard = graph_->create(prim::Guard, {n->input()}, 1);
          auto go = guard->output();
          auto copy = ProfiledTensorType::create(pttp);
          go->setType(copy);
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

} // namespace jit
} // namespace torch
