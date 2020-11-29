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

TEST(SubgraphUtilsTest, Vmap) {
  auto graph = std::make_shared<Graph>();

  std::unordered_map<std::string, Value*> parse_map;
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::tanh(%a)
  %y : Tensor = aten::mul(%a, %b)

  %p : Tensor = aten::div(%c, %b)
  %q : Tensor = aten::mul(%p, %a)
  return (%x, %y, %q))IR",
      &*graph,
      parse_map);

  std::unordered_map<Value*, Value*> vmap1;

  Node* subgraph1 = SubgraphUtils::createSingletonSubgraph(
      parse_map.at("y")->node(), prim::DifferentiableGraph);
  SubgraphUtils::mergeNodeIntoSubgraph(
      parse_map.at("x")->node(), subgraph1, vmap1);
  // vmap should have two entries: a mapping for the '%x' value - the output of
  // the node we merged in, and a mapping for the '%a' value - the input of the
  // node.
  ASSERT_EQ(vmap1.size(), 2);

  // Check that after mergeNodeIntoSubgraph we can still access the node
  // corresponding to the original node "%x = aten::tanh(%a)".
  //
  // Note that parse_map["x"] points to a destroyed Value after the
  // mergeNodeIntoSubgraph call - we cannot access its content anymore, but
  // still can use it as a key in the value map to find the value it was moved
  // to.
  Node* new_tanh = vmap1.at(parse_map.at("x"))->node();
  ASSERT_TRUE(new_tanh->kind() == aten::tanh);

  Node* subgraph2 = SubgraphUtils::createSingletonSubgraph(
      parse_map["q"]->node(), prim::DifferentiableGraph);
  SubgraphUtils::mergeNodeIntoSubgraph(parse_map.at("p")->node(), subgraph2);

  std::unordered_map<Value*, Value*> vmap2;
  Value* new_tanh_out = new_tanh->output();
  SubgraphUtils::mergeNodeIntoSubgraph(subgraph1, subgraph2, vmap2);
  // vmap should have 6 entries, since we moved 4 values into the graph (the
  // values correspond to the original values '%a', '%b', '%x', and '%y').
  // and we map the node outputs for '%x' and '%y'
  ASSERT_EQ(vmap2.size(), 6);

  // Check that after mergeNodeIntoSubgraph we can still access the node
  // corresponding to the original node, even if the toMerge node had a subgraph
  // as well
  ASSERT_TRUE(vmap2.at(new_tanh_out)->node()->kind() == aten::tanh);
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
