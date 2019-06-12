#include "torch/csrc/jit/passes/inline_autodiff_subgraphs.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

namespace {

bool canRunWithAutograd(Node *node) {
  return node->kind() != prim::FusionGroup;
}

void inlineNode(Node *node) {
  WithInsertPoint insert_guard { node };
  Graph * graph = node->owningGraph();
  auto subgraph = node->g(attr::Subgraph);
  std::unordered_map<Value*, Value*> input_map;

  size_t num_inputs = node->inputs().size();
  JIT_ASSERT(num_inputs == subgraph->inputs().size());
  for (size_t i = 0; i < num_inputs; ++i) {
    input_map[subgraph->inputs()[i]] = node->inputs()[i];
  }

  for (Node * subnode : subgraph->nodes()) {
    Node * new_node = graph->insertNode(graph->createClone(subnode, [&](Value * v) { return input_map.at(v); }));
    for (size_t i = 0; i < subnode->outputs().size(); ++i) {
      input_map[subnode->output(i)] = new_node->output(i);
    }
  }

  size_t num_outputs = node->outputs().size();
  JIT_ASSERT(num_outputs <= subgraph->outputs().size() &&
             num_outputs == static_cast<size_t>(node->i(attr::f_real_outputs)));
  for (size_t i = 0; i < num_outputs; ++i) {
    node->output(i)->replaceAllUsesWith(input_map.at(subgraph->outputs()[i]));
  }
}

void InlineAutodiffSubgraphs(Block *block, size_t threshold) {
  for (Node * node : block->nodes()) {
    for (Block * block : node->blocks()) {
      InlineAutodiffSubgraphs(block, threshold);
    }
    if (node->kind() != prim::DifferentiableGraph) continue;
    auto subgraph = node->g(attr::Subgraph);
    int64_t subgraph_size = std::distance(subgraph->nodes().begin(), subgraph->nodes().end());
    if (subgraph_size >= static_cast<int64_t>(threshold)) continue;
    if (!std::all_of(subgraph->nodes().begin(), subgraph->nodes().end(), canRunWithAutograd)) continue;
    inlineNode(node);
  }
}

} // anonymous namespace

void InlineAutodiffSubgraphs(std::shared_ptr<Graph>& graph, size_t threshold) {
  InlineAutodiffSubgraphs(graph->block(), threshold);
  EliminateDeadCode(graph);
}

}} // namespace torch::jit
