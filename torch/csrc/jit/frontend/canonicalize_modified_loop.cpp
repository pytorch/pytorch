#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>

namespace torch::jit {

// Transforms a Loop that has both a trip count specified and a loop
// body condition so that the iter count is no longer specified
// and it is recognizable as a python while loop.
void canonicalizeModifiedLoop(Node* n) {
  LoopView loop(n);
  if (loop.loopType() != LoopView::ModifiedLoop) {
    return;
  }

  auto g = n->owningGraph();
  WithInsertPoint node_insert(n);
  auto zero = g->insertConstant(0);
  auto one = g->insertConstant(1);
  auto max_trip_count = loop.maxTripCount();
  auto condition = g->insert(aten::gt, {max_trip_count, zero});
  loop.replaceMaxTripCount(
      g->insertConstant(std::numeric_limits<int64_t>::max()));

  auto inp_condition = toIValue(loop.inputCond());
  if (inp_condition == c10::nullopt || inp_condition->toBool() == false) {
    condition = g->insert(aten::__and__, {condition, loop.inputCond()});
  }
  loop.replaceInputCondition(condition);
  n->addOutput()->setType(IntType::get());
  WithInsertPoint loop_insert(loop.bodyBlock());
  n->addInput(zero);
  auto new_iter = loop.bodyBlock()->addInput()->setType(IntType::get());
  // unset unique name for jitter, its replacement does not have a name
  loop.currentTripCount()->setDebugName("")->replaceAllUsesWith(new_iter);
  auto inc_iter = g->insert(aten::add, {new_iter, one});
  loop.bodyBlock()->registerOutput(inc_iter);
  auto less_than_max_trip = g->insert(aten::lt, {inc_iter, max_trip_count});
  auto loop_continue = loop.nextCond();
  auto new_condition =
      g->insert(aten::__and__, {less_than_max_trip, loop_continue});
  loop.bodyBlock()->eraseOutput(0);
  loop.bodyBlock()->insertOutput(0, new_condition);
}

void canonicalizeModifiedLoops(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      canonicalizeModifiedLoops(b);
    }
    if (n->kind() == prim::Loop) {
      canonicalizeModifiedLoop(n);
    }
  }
}

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph) {
  canonicalizeModifiedLoops(graph->block());
}

} // namespace torch::jit
