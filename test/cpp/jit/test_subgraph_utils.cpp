#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"

namespace torch {
namespace jit {

void testSubgraphUtils() {
  auto graph = build_lstm();
  EliminateCommonSubexpression(graph);

  std::vector<Node*> originalNodes(
      graph->nodes().begin(), graph->nodes().end());

  // Merge everything into a single subgraph
  bool first = true;
  Node* subgraph;
  for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend();) {
    if (first) {
      subgraph = SubgraphUtils::createSingletonSubgraph(
          *it, prim::DifferentiableGraph);
      it = ++subgraph->reverseIterator();
      first = false;
    }

    SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph);
    it = ++subgraph->reverseIterator();
  }

  // Unmerge and compare with original node listing
  SubgraphUtils::unmergeSubgraph(subgraph);
  EliminateCommonSubexpression(graph);

  std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
  ASSERT_EQ(originalNodes.size(), newNodes.size());
}

} // namespace jit
} // namespace torch
