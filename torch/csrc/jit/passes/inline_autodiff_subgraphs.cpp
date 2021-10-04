#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

// aten and prim nodes (except FusionGroup) are guaranteed to work
// with Autograd, other nodes (e.g. user-defined nodes) are not necessarily
// Autograd-aware
bool canRunWithAutograd(Node* node) {
  auto kind = node->kind();
  for (Block* block : node->blocks()) {
    if (!std::all_of(
            block->nodes().begin(), block->nodes().end(), canRunWithAutograd)) {
      return false;
    }
  }
  return kind != prim::FusionGroup && kind != prim::CudaFusionGroup &&
      kind != prim::TypeCheck && kind != prim::TensorExprGroup &&
      kind != prim::CudaFusionGuard && (kind.is_aten() || kind.is_prim());
}

namespace {

void InlineAutodiffSubgraphs(Block* block, size_t threshold);

size_t blockSize(Block* block) {
  size_t num = 0;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      num += blockSize(b);
    }
    num++;
  }
  return num;
}

graph_node_list::iterator scanNode(Node* node, size_t threshold) {
  auto next_node = ++node->iterator();

  for (Block* block : node->blocks()) {
    InlineAutodiffSubgraphs(block, threshold);
  }

  if (node->kind() != prim::DifferentiableGraph) {
    return next_node;
  }

  auto subgraph = node->g(attr::Subgraph);
  size_t subgraph_size = blockSize(subgraph->block());
  if (subgraph_size >= threshold) {
    return next_node;
  }

  if (!std::all_of(
          subgraph->nodes().begin(),
          subgraph->nodes().end(),
          canRunWithAutograd)) {
    return next_node;
  }

  // now that we inline the graph, we are no longer detaching input tensors,
  // so the profiles will have outdated requires_grad=False.
  // conservatively update them to maybe requiring grad, bc we might create
  // autodiff graphs when the tensors maybe require grad
  UpdateDifferentiableGraphRequiresGrad(subgraph, c10::nullopt);
  SubgraphUtils::unmergeSubgraph(node);
  return next_node;
}

void InlineAutodiffSubgraphs(Block* block, size_t threshold) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    it = scanNode(*it, threshold);
  }
}

} // anonymous namespace

void InlineAutodiffSubgraphs(std::shared_ptr<Graph>& graph, size_t threshold) {
  InlineAutodiffSubgraphs(graph->block(), threshold);
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
