#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"

#include <torch/csrc/jit/testing/file_check.h>
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SubgraphUtilsTest, Basic) {
  auto graph = build_lstm();
  EliminateCommonSubexpression(graph);

  std::vector<Node*> originalNodes(
      graph->nodes().begin(), graph->nodes().end());

  for (bool reverse_iterate : {true, false}) {
    // Merge everything into a single subgraph
    bool first = true;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* subgraph;
    auto it =
        reverse_iterate ? graph->nodes().rbegin() : graph->nodes().begin();
    auto end = reverse_iterate ? graph->nodes().rend() : graph->nodes().end();
    for (; it != end;) {
      if (first) {
        subgraph = SubgraphUtils::createSingletonSubgraph(
            *it, prim::DifferentiableGraph);
        it = reverse_iterate ? ++subgraph->reverseIterator()
                             : ++subgraph->iterator();
        first = false;
      }

      SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph);
      it = reverse_iterate ? ++subgraph->reverseIterator()
                           : ++subgraph->iterator();
    }

    // Unmerge and compare with original node listing
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    SubgraphUtils::unmergeSubgraph(subgraph);
    EliminateCommonSubexpression(graph);

    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
    ASSERT_EQ(originalNodes.size(), newNodes.size());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SubgraphUtilsTest, MergeSubgraphs) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> parse_map;
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::sigmoid(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::hardsigmoid(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);

  std::vector<Node*> originalNodes(
      graph->nodes().begin(), graph->nodes().end());
  for (bool reverse_merge : {true, false}) {
    // Merge everything into two adjacent subgraphs
    Node* graph1 = SubgraphUtils::createSingletonSubgraph(
        *graph->nodes().begin(), prim::DifferentiableGraph);
    while (true) {
      Node* next = graph1->next();
      if (next->kind() == aten::tanh) {
        break;
      }
      SubgraphUtils::mergeNodeIntoSubgraph(next, graph1);
    }
    Node* graph2 = SubgraphUtils::createSingletonSubgraph(
        graph1->next(), prim::DifferentiableGraph);
    while (graph2->next() != *graph->nodes().end()) {
      SubgraphUtils::mergeNodeIntoSubgraph(graph2->next(), graph2);
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* subgraph;
    if (reverse_merge) {
      SubgraphUtils::mergeNodeIntoSubgraph(graph2, graph1);
      subgraph = graph1;
    } else {
      SubgraphUtils::mergeNodeIntoSubgraph(graph1, graph2);
      subgraph = graph2;
    }
    auto run_file_check = [](std::shared_ptr<Graph> graph) {
      graph->lint();
      testing::FileCheck()
          .check("aten::sigmoid")
          ->check("aten::mul")
          ->check("aten::div")
          ->check("aten::mul")
          ->check_count("aten::tanh", 3)
          ->check("aten::hardsigmoid")
          ->run(*graph);
    };
    run_file_check(subgraph->g(attr::Subgraph));

    // Unmerge and compare with original node listing
    SubgraphUtils::unmergeSubgraph(subgraph);
    EliminateCommonSubexpression(graph);
    run_file_check(graph);

    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
    ASSERT_EQ(originalNodes.size(), newNodes.size());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
