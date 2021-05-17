#include <algorithm>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/linear_to_matmul_add.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

namespace torch {
namespace jit {

namespace {

void DecomposeLinearToMatmulAdd(Block *b) {
  // TODO only do the expansion for GPU since CPU fusion isn't on by default
  // (and thus may slow down instead)
  std::vector<Node*> linear_nodes;
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    for (auto* child_block : it->blocks()) {
      DecomposeLinearToMatmulAdd(child_block);
    }
    if (it->kind() == aten::linear) {
      linear_nodes.push_back(*it);
    }
  }

  for (Node* node : linear_nodes) {
    // If all its uses are supported for fusion, decompose into matmul + add
    auto uses = node->output()->uses();
    if (std::all_of(uses.begin(), uses.end(), [](const Use& u) {
          return tensorexpr::isSupported(u.user);
        })) {
      auto g = b->owningGraph();
      auto matmul_n =
          g->create(
               aten::matmul,
               {node->namedInput("input"), node->namedInput("weight")},
               1)
              ->insertBefore(node);
      auto add_n =
          g->create(
               aten::add, {matmul_n->output(), node->namedInput("bias")}, 1)
              ->insertBefore(node);
      // TODO aten::linear says bias is optional: what about that case?
      node->output()->replaceAllUsesWith(add_n->output());
      node->destroy();
    }
  }
}

} // namespace

void DecomposeLinearToMatmulAdd(std::shared_ptr<Graph>& graph) {
  DecomposeLinearToMatmulAdd(graph->block());
}

} // namespace jit
} // namespace torch
