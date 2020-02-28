#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

// aten and prim nodes (except FusionGroup) are guaranteed to work
// with Autograd, other nodes (e.g. user-defined nodes) are not necessarily
// Autograd-aware
bool canRunWithAutograd(Node* node) {
  auto kind = node->kind();
  return kind != prim::FusionGroup && kind != prim::CudaFusionGroup &&
         (kind.is_aten() || kind.is_prim());
}

namespace {

void InlineAutodiffSubgraphs(Block* block, size_t threshold);

graph_node_list::iterator scanNode(Node* node, size_t threshold) {
  auto next_node = ++node->iterator();

  for (Block* block : node->blocks()) {
    InlineAutodiffSubgraphs(block, threshold);
  }

  if (node->kind() != prim::DifferentiableGraph) {
    return next_node;
  }

  auto subgraph = node->g(attr::Subgraph);
  int64_t subgraph_size =
      std::distance(subgraph->nodes().begin(), subgraph->nodes().end());
  if (subgraph_size >= static_cast<int64_t>(threshold)) {
    return next_node;
  }

  if (!std::all_of(
          subgraph->nodes().begin(),
          subgraph->nodes().end(),
          canRunWithAutograd)) {
    return next_node;
  }

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
