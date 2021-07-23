#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
using namespace testing;

TEST(SubgraphRewriterTest, FilterMatch) {
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

TEST(SubgraphRewriterTest, FilterNoMatch) {
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

TEST(SubgraphRewriterTest, MultiOutput) {
  {
    auto graph = std::make_shared<Graph>();

    // Basic multi-output pattern rewriting
    parseIR(
        R"IR(
graph(%0, %1):
  %a1, %a2 = a::aaa(%0, %1)
  %b = b::bbb(%a1)
  %c = c::ccc(%b)

  %x1, %x2 = a::aaa(%c, %a2)
  %y = b::bbb(%x1)
  %z = d::ddd(%y)
  return (%z))IR",
        graph.get());

    std::string pattern = R"IR(
graph(%0, %1):
  %a1, %a2 = a::aaa(%0, %1)
  %b = b::bbb(%a1)
  return (%b, %a2))IR";

    std::string replacement = R"IR(
graph(%a, %b):
  %x, %y = ab::ababab(%a, %b)
  return (%x, %y))IR";

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(pattern, replacement);

    auto g = graph->copy();
    rewriter.runOnGraph(g);
    FileCheck().check("ab::ababab")->check("ab::ababab")->run(*g);
  }
  {
    auto graph = std::make_shared<Graph>();

    // Mimic a real model case
    parseIR(
        R"IR(
    graph(%k, %m, %x1, %x2, %x3, %x4, %y1, %y2, %y3, %y4):
      %a1 = aa::aaa(%x1, %k)
      %b1_1, %b1_2 = bb::bbb(%y1, %a1)
      %a2 = aa::aaa(%x2, %k)
      %b2_1, %b2_2 = bb::bbb(%y2, %a2)
      %a3 = aa::aaa(%x3, %k)
      %b3_1, %b3_2 = bb::bbb(%y3, %a3)
      %a4 = aa::aaa(%x4, %k)
      %b4_1, %b4_2 = bb::bbb(%y4, %a4)
      %c = cc::ccc(%b4_1)
      %d1 = dd::ddd(%b1_2, %m)
      %e1 = ee::eee(%b1_1, %d1)
      %d2 = dd::ddd(%b2_2, %m)
      %e2 = ee::eee(%b2_1, %d2)
      %d3 = dd::ddd(%b3_2, %m)
      %e3 = ee::eee(%b3_1, %d3)
      %d4 = dd::ddd(%b4_2, %m)
      %e4 = ee::eee(%b4_1, %d4)
      return (%d1, %d2, %d3, %d4, %e1, %e2, %e3, %e4)
      )IR",
        graph.get());

    std::string pattern = R"IR(
    graph(%a, %b, %c, %d):
        %y0 = aa::aaa(%b, %c)
        %y1, %y2 = bb::bbb(%a, %y0)
        %y3 = dd::ddd(%y2, %d)
        return (%y3, %y1))IR";

    std::string replacement = R"IR(
    graph(%a, %b, %c, %d):
      %x, %y = ab::ababab(%a, %b, %c, %d)
      return (%x, %y))IR";

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(pattern, replacement);

    auto g = graph->copy();
    rewriter.runOnGraph(g);
    FileCheck().check("ab::ababab")->check("ab::ababab")->run(*g);
  }
  {
    auto graph = std::make_shared<Graph>();

    // A case where no rewriting should occur due to data dependencies
    parseIR(
        R"IR(
    graph(%x, %y):
      %a = aa::aaa(%x)
      %b = bb::bbb(%a)
      %e = ee::eee(%b)
      %c = cc::ccc(%y)
      %d = dd::ddd(%b, %c)
      %f = ff::fff(%b, %d)
      return (%f)
      )IR",
        graph.get());

    std::string pattern = R"IR(
    graph(%a, %c):
        %b = bb::bbb(%a)
        %d = dd::ddd(%b, %c)
        return (%d, %b))IR";

    std::string replacement = R"IR(
    graph(%a, %c):
      %d, %b = db::fused(%a, %c)
      return (%d, %b))IR";

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(pattern, replacement);

    auto g = graph->copy();
    rewriter.runOnGraph(g);
    // We should not perform the replacement on the given graph due to data
    // dependency constraints: the output %b is used in %e, which precedes one
    // def of the input %c.
    FileCheck().check_not("db::fused")->run(*g);
  }
}
} // namespace jit
} // namespace torch
