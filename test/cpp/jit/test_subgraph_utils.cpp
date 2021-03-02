#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"

namespace torch {
namespace jit {

TEST(SubgraphUtilsTest, Basic) {
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

TEST(SubgraphUtilsTest, GraphName) {
  auto graph = std::make_shared<Graph>();

  std::unordered_map<std::string, Value*> parse_map;
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::tanh(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::tanh(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);
  std::string ref_full_name = "graph_tanh_mul_div_mul_tanh_tanh_tanh_tanh";
  std::string full_name =
      SubgraphUtils::generateNameForGraph(graph, 80, "graph");
  ASSERT_EQ(full_name, ref_full_name);

  std::string truncated_name =
      SubgraphUtils::generateNameForGraph(graph, 10, "graph");

  ASSERT_LE(truncated_name.size(), ref_full_name.size());
}

} // namespace jit
} // namespace torch
