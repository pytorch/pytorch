#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

struct GuardElimination {
  GuardElimination(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)),
        aliasDb_(caffe2::make_unique<AliasDb>(graph_)) {}

  void run() {
    moveGuardsToDefs(graph_->block());
    coalesceGuards(graph_->block());
    eliminateGuards(graph_->block());
  }

  void moveGuardsToDefs(Block* b) {
    // alias analysis gets confused if we ask it to move expressions
    // around prim::load, so we insert a dummy anchor to
    // which we will be moving uses of params
    // that alias analysis tells us are okay to move
    auto start_node = b->owningGraph()->create(prim::Constant, 1);
    start_node->output()->setType(IntType::create());
    // the answer to the mystery of life
    start_node->i_(attr::value, 42);
    start_node->insertAfter(*b->nodes().begin());
    for (auto it = b->nodes().begin(); it != b->nodes().end();) {
      auto n = *it;
      if (n->kind() == prim::Guard) {
        // grab the next node before we move this one all the way back
        it++;
        auto guardee = n->inputs().at(0)->node();
        if (guardee->kind() == prim::Param) {
          guardee = start_node;
        }
        aliasDb_->moveAfterTopologicallyValid(n, guardee);
      } else {
        it++;
        for (Block* ib : n->blocks()) {
          moveGuardsToDefs(ib);
        }
      }
    }
  }

  void coalesceGuards(Block* b) {
    // uses on *all* parameters are moved to the same anchor node
    // and they may come in different order after the anchor node
    // e.g. (anchor, guard_x, guard_y, guard_x, guard_y)
    // this pass recognizes contigious streches of guards and
    // keeps track of the guards it's seen for each def. the next time
    // the guard on the same def, it simply removes it.
    std::unordered_map<Value*, Node*> inputs_to_guards;
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::Guard) {
        if (inputs_to_guards.count(n->input())) {
          auto prev = inputs_to_guards[n->input()];
          n->output()->replaceAllUsesWith(prev->output());
          it.destroyCurrent();
        } else {
          inputs_to_guards.insert({n->input(), n});
        }
      } else {
        inputs_to_guards.clear();
        for (Block* ib : n->blocks()) {
          moveGuardsToDefs(ib);
        }
      }
    }
  }

  void eliminateGuards(Block* b) {
    // a very simple pass to eliminate redundant guards for ops
    // whose outputs are fully determined by their inputs
    // i.e. if inputs to such ops are guarded we are allowed
    // to remove a guard on ops' outputs
    for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
      auto n = *it;
      if (n->kind() == prim::Guard &&
          removableGuard(n->inputs().at(0)->node())) {
        auto pttp = n->output()->type();
        n->output()->replaceAllUsesWith(n->inputs().at(0));
        n->inputs().at(0)->setType(pttp);
        it.destroyCurrent();
      } else {
        it++;
        for (Block* ib : n->blocks()) {
          moveGuardsToDefs(ib);
        }
      }
    }
  }

 private:
  bool removableGuard(Node* n) {
    if (!simple_ops_.count(n->kind())) {
      return false;
    }

    bool all_inputs_guarded = true;
    for (auto input : n->inputs()) {
      if (input->node()->kind() == prim::Guard ||
          input->node()->kind() == prim::Constant) {
        AT_ASSERT(
            input->node()->kind() != prim::Guard ||
            input->type()->expect<ProfiledTensorType>());
      } else {
        all_inputs_guarded = false;
        break;
      }
    }
    return all_inputs_guarded;
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
  static std::unordered_set<Symbol> simple_ops_;
};

std::unordered_set<Symbol> GuardElimination::simple_ops_ = {aten::add,
                                                            aten::sub,
                                                            aten::mul,
                                                            aten::div};

static void removeProfilingNodes(Block* b) {
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

void EliminateGuards(std::shared_ptr<Graph> graph) {
  GuardElimination ge(std::move(graph));
  ge.run();
}

} // namespace jit
} // namespace torch
