#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/assertions.h"

namespace torch { namespace jit {

static void LowerTuples(Block* block);

static void VisitNode(Node* n) {
  // make any TupleUnpack dead by undoing TupleUnpack(TupleConstruct())
  if(n->kind() == prim::TupleUnpack) {
    auto construct = n->input()->node();
    if(construct->kind() == prim::TupleConstruct) {
      for(size_t i = 0; i < n->outputs().size(); ++i) {
        n->outputs()[i]->replaceAllUsesWith(construct->inputs()[i]);
      }
      // op is now dead let dce clean up
      return;
    }
  }

  for(auto b : n->blocks()) {
    LowerTuples(b);
  }
}

static void LowerTuples(Block* block) {
  for(auto n : block->nodes()) {
    VisitNode(n);
  }
}

void LowerTuples(std::shared_ptr<Graph>& graph) {
  LowerTuples(graph->block());
  EliminateDeadCode(graph);
}

}}
