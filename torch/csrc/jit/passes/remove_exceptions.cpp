#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/remove_exceptions.h>

#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

bool certainlyThrows(Block* block) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::RaiseException) {
      return true;
    }
  }
  return false;
}

void EliminateExceptions(Block* block) {
  auto graph = block->owningGraph();
  Value* false_const = graph->insertConstant(IValue(false));
  Value* true_const = graph->insertConstant(IValue(true));
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::If) {
      Block* true_block = n->blocks()[0];
      Block* false_block = n->blocks()[1];
      if (certainlyThrows(true_block)) {
        n->input(0)->replaceAllUsesWith(false_const);
      } else if (certainlyThrows(false_block)) {
        n->input(0)->replaceAllUsesWith(true_const);
      }
    }

    for (Block* subblock : n->blocks()) {
      EliminateExceptions(subblock);
    }
  }
}

void EliminateExceptions(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before EliminateExceptions: ", graph);
  EliminateExceptions(graph->block());
  ConstantPropagation(graph);
  ConstantPooling(graph);
  GRAPH_DUMP("After EliminateExceptions: ", graph);
}

} // namespace jit
} // namespace torch
