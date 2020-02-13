#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
using namespace testing;

void testFilterMatch() {
  auto graph = std::make_shared<Graph>();

  script::parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = prim::Constant[value=1]()
  %c = c::ccc(%a, %b)
  return (%c))IR",
      graph.get());

  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;

  script::parseIR(
      pattern,
      &pattern_graph,
      vmap);

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
     const auto& match_vmap = match.values_map;
     auto b_node = match_vmap.at(vmap.at("b"))->node();
     return b_node->kind() == prim::Constant;
   };

  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(pattern, replacement);
  rewriter.runOnGraph(graph, filter);

  FileCheck().check("d::ddd")
    ->check_not("c::ccc")
    ->run(*graph);
}

void testFilterNoMatch() {
  auto graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = prim::Constant[value=1]()
  %c = c::ccc(%a, %b)
  return (%c))IR",
      graph.get());

  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;

  script::parseIR(
      pattern,
      &pattern_graph,
      vmap);

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
     const auto& match_vmap = match.values_map;
     auto b_node = match_vmap.at(vmap.at("b"))->node();
     // b_node is not Constant, so this won't match and we'll skip the rewrite
     return b_node->kind() == prim::Assign;
   };

  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(pattern, replacement);
  rewriter.runOnGraph(graph, filter);

  FileCheck().check("c::ccc")
    ->check_not("d::ddd")
    ->run(*graph);

}


void testSubgraphRewriter() {
  testFilterMatch();
  testFilterNoMatch();
}

}}
