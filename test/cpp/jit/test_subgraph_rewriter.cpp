#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
using namespace testing;

void testFilterMatch() {
  auto graph = std::make_shared<Graph>();

  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b : int = prim::Constant[value=1]()
  %c = c::ccc(%a, %b)
  return (%c))IR",
      graph.get());

  std::string pattern = R"IR(
graph(%a, %b):
  %c = c::ccc(%a, %b)
  return (%c))IR";
  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;

  parseIR(pattern, &pattern_graph, vmap);

  auto b_is_constant = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_node = match_vmap.at(vmap.at("b"))->node();
    return b_node->kind() == prim::Constant;
  };

  auto b_is_one = [](const Match& match,
                     const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_val = toIValue(match_vmap.at(vmap.at("b")));
    return b_val && b_val->isInt() && b_val->toInt() == 1;
  };

  auto b_is_two = [](const Match& match,
                     const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_val = toIValue(match_vmap.at(vmap.at("b")));
    return b_val && b_val->isInt() && b_val->toInt() == 2;
  };

  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(pattern, replacement);

  // b is constant, so the match will succeed
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, b_is_constant);
    FileCheck().check("d::ddd")->check_not("c::ccc")->run(*g);
  }

  // b is constant and the value is one, the match will succeed
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, {b_is_constant, b_is_one});
    FileCheck().check("d::ddd")->check_not("c::ccc")->run(*g);
  }

  // b is constant but the value is not two, the match will fail
  {
    auto g = graph->copy();
    rewriter.runOnGraph(g, {b_is_constant, b_is_two});
    FileCheck().check("c::ccc")->check_not("d::ddd")->run(*g);
  }
}

void testFilterNoMatch() {
  auto graph = std::make_shared<Graph>();
  parseIR(
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

  parseIR(pattern, &pattern_graph, vmap);

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto b_node = match_vmap.at(vmap.at("b"))->node();
    // b_node is not prim::Assign, so this won't match and we'll skip the
    // rewrite
    return b_node->kind() == prim::Assign;
  };

  std::string replacement = R"IR(
graph(%a, %b):
  %d = d::ddd(%a, %b)
  return (%d))IR";

  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(pattern, replacement);
  rewriter.runOnGraph(graph, filter);

  FileCheck().check("c::ccc")->check_not("d::ddd")->run(*graph);
}

void testSubgraphRewriter() {
  testFilterMatch();
  testFilterNoMatch();
}

} // namespace jit
} // namespace torch
