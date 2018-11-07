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

void EliminateDeadCode(Block *block, bool recurse, bool_memo_type& memo) {
  auto nodes = block->nodes().reverse();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
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
