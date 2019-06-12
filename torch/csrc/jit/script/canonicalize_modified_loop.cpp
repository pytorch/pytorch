#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/script/canonicalize_modified_loop.h>

namespace torch {
namespace jit {
namespace script {

// Transforms a Loop that has both a trip count specified and a loop
// body condition so that the iter count is no longer specified
// and it is recognizable as a python while loop.
void canonicalizeModifiedLoops(Node* n) {
  LoopView loop(n);
  if (loop.loopType() != LoopView::ModifiedLoop) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      constant_as<bool>(n->inputs().at(1)).value_or(false) == true,
      "non-constant loop input condition NYI");

  auto g = n->owningGraph();
  WithInsertPoint node_insert(n);
  auto zero = g->insertConstant(0);
  auto one = g->insertConstant(1);
  auto max_trip_count = loop.maxTripCount();
  auto condition = g->insert(aten::gt, {max_trip_count, zero});
  n->removeInput(0);
  n->insertInput(0, g->insertConstant(std::numeric_limits<int64_t>::max()));
  n->removeInput(1);
  n->insertInput(1, condition);
  n->addOutput()->setType(IntType::get());
  WithInsertPoint loop_insert(loop.bodyBlock());
  n->addInput(zero);
  auto new_iter = loop.bodyBlock()->addInput()->setType(IntType::get());
  // unset unique name for jitter, its replacement does not have a name
  loop.currentTripCount()->setUniqueName("")->replaceAllUsesWith(new_iter);
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
    switch (n->kind()) {
      case prim::Function:
      case prim::If: {
        for (Block* b : n->blocks()) {
          canonicalizeModifiedLoops(b);
        }
      } break;
      case prim::Loop: {
        canonicalizeModifiedLoops(n->blocks().at(0));
        canonicalizeModifiedLoops(n);
      } break;
    }
  }
}

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph) {
  canonicalizeModifiedLoops(graph->block());
}

} // namespace script
} // namespace jit
} // namespace torch
