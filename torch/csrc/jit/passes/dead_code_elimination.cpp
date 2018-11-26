#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include <unordered_map>

namespace torch { namespace jit {

using bool_memo_type = std::unordered_map<Node*, bool>;

bool isMutable(Node* node) {
  if(!node->kind().is_aten())
    return false;
  // onnx export calls EliminateDeadCode but sometimes passes invalid
  // aten operators. So we call maybeSchema so we handle the cases when
  // there is no valid schema for a node
  auto schema = node->maybeSchema();
  return schema && schema->is_mutable();
}

bool hasSideEffects(Node * node, bool_memo_type& memo) {
  // FIXME: PythonOp should be treated as having side effects as well!
  //        Unfortunately ONNX depends on it getting removed in this pass, so it's not
  //        a simple change.
  auto it = memo.find(node);
  if (it != memo.end())
    return it->second;
  bool has_side_effects =
      node->kind() == prim::Print ||
        node->kind() == prim::RaiseException ||
      std::any_of(node->blocks().begin(), node->blocks().end(), [&](Block* b) {
        return std::any_of(b->nodes().begin(), b->nodes().end(), [&](Node* n) {
          return hasSideEffects(n, memo);
        });
      }) || isMutable(node);

  memo.emplace(node, has_side_effects);
  return has_side_effects;
}

void removeDeadIfOutputs(Node* node) {
  if (node->kind() != prim::If)
    return;
  for(size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!node->outputs().at(i)->hasUses()) {
      node->eraseOutput(i);
      for (Block* b : node->blocks()) {
        b->eraseOutput(i);
      }
    }
  }
}

void removeDeadLoopOutputs(Node* node) {
  if (node->kind() != prim::Loop)
    return;
  auto loop_body = node->blocks().at(0);
  auto loop_input_offset = 2; // offset of loop carried deps in input list
  auto loop_body_offset = 1; // offset to the loop carried dependencies in block inputs/outputs

  for(size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!node->outputs().at(i)->hasUses() &&
        !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
      node->eraseOutput(i);
      node->removeInput(loop_input_offset + i);
      loop_body->eraseInput(loop_body_offset + i);
      loop_body->eraseOutput(loop_body_offset + i);
    }
  }
}

void EliminateDeadCode(Block *block, bool recurse, bool_memo_type& memo) {
  auto nodes = block->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    // note these occur before the recursion because we want to uncover
    // dead code in the blocks used to calculate the output
    removeDeadIfOutputs(node);
    removeDeadLoopOutputs(node);
    if (recurse) {
      for (Block * block : node->blocks())
        EliminateDeadCode(block, true, memo);
    }
    if (!node->hasUses() && !hasSideEffects(node, memo))
      it.destroyCurrent();
  }
}

void EliminateDeadCode(const std::shared_ptr<Graph>& graph) {
  bool_memo_type side_effect_memo;
  EliminateDeadCode(graph->block(), true, side_effect_memo);
}

void EliminateDeadCode(Block *block, bool recurse) {
  bool_memo_type side_effect_memo;
  EliminateDeadCode(block, recurse, side_effect_memo);
}

}} // namespace torch::jit
