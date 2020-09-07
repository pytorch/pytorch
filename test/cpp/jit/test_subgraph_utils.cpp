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

void testSubgraphUtilsVmap() {
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

} // namespace jit
} // namespace torch
