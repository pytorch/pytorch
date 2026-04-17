#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"

namespace torch {
namespace jit {

TEST(SubgraphMatcherTest, Trivial1) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  return (%a))IR",
      &graph);
  parseIR(
      R"IR(
graph(%0):
  %x = a::aaa(%0)
  return (%x))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

TEST(SubgraphMatcherTest, Trivial2) {
  Graph graph;
  auto* g_in = graph.addInput();
  auto* g_tanh = graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  g_tanh->addInput(g_in);
  graph.registerOutput(g_tanh->output());

  Graph pattern;
  auto* p_in = pattern.addInput();
  auto* p_tanh =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  p_tanh->addInput(p_in);
  pattern.registerOutput(p_tanh->output());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    AT_ASSERT(m.values_map.at(p_tanh->output()) == g_tanh->output());
    AT_ASSERT(m.nodes_map.at(p_tanh) == g_tanh);
  }
}

TEST(SubgraphMatcherTest, Trivial3) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%0):
  %a = a::a(%0)
  %b = a::b(%0)
  %c = a::c(%a, %b)
  return (%c))IR",
      &graph);
  parseIR(
      R"IR(
graph(%a, %b):
  %c = a::c(%a, %b)
  return (%c))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

TEST(SubgraphMatcherTest, Trivial4) {
  Graph graph;
  auto* g_in0 = graph.addInput();
  auto* g_in1 = graph.addInput();
  auto* g_mul = graph.insertNode(graph.create(aten::mul, /*num_outputs =*/1));
  g_mul->addInput(g_in0);
  g_mul->addInput(g_in1);
  graph.registerOutput(g_mul->output());

  Graph pattern;
  auto* p_in0 = pattern.addInput();
  auto* p_in1 = pattern.addInput();
  auto* p_mul =
      pattern.insertNode(pattern.create(aten::mul, /*num_outputs =*/1));
  p_mul->addInput(p_in0);
  p_mul->addInput(p_in1);
  pattern.registerOutput(p_mul->output());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in0) == g_in0);
    AT_ASSERT(m.values_map.at(p_in1) == g_in1);
    AT_ASSERT(m.values_map.at(p_mul->output()) == g_mul->output());
    AT_ASSERT(m.nodes_map.at(p_mul) == g_mul);
  }
}

TEST(SubgraphMatcherTest, Linear1) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%b)
  %d = d::ddd(%c)
  %a = a::aaa(%0)
  return (%d))IR",
      &graph);
  parseIR(
      R"IR(
graph(%0):
  %x = b::bbb(%0)
  %y = c::ccc(%x)
  return (%y))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

TEST(SubgraphMatcherTest, Linear2) {
  Graph graph;
  auto* g_in = graph.addInput();

  auto* g_tanh = graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  g_tanh->addInput(g_in);

  auto* g_tanh2 =
      graph.insertNode(graph.create(aten::tanh, /*num_outputs =*/1));
  g_tanh2->addInput(g_tanh->output());

  graph.registerOutput(g_tanh2->output());

  Graph pattern;
  auto* p_in = pattern.addInput();

  auto* p_tanh =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  p_tanh->addInput(p_in);

  auto* p_tanh2 =
      pattern.insertNode(pattern.create(aten::tanh, /*num_outputs =*/1));
  p_tanh2->addInput(p_tanh->output());

  pattern.registerOutput(p_tanh2->output());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    AT_ASSERT(m.values_map.at(p_tanh->output()) == g_tanh->output());
    AT_ASSERT(m.values_map.at(p_tanh2->output()) == g_tanh2->output());
    AT_ASSERT(m.nodes_map.at(p_tanh) == g_tanh);
    AT_ASSERT(m.nodes_map.at(p_tanh2) == g_tanh2);
  }
}

/**
 * Test diamond pattern:
 *
 *     ooo
 *      |
 *     aaa
 *    /   \
 *  bbb   ccc
 *     \ /
 *     ddd
 *      |
 *     eee
 */
TEST(SubgraphMatcherTest, Diamond1) {
  Graph graph, pattern1, pattern2;
  parseIR(
      R"IR(
graph(%0):
  %o = o::ooo(%0)
  %a = a::aaa(%o)
  %b = b::bbb(%a)
  %c = c::ccc(%a)
  %d = d::ddd(%b, %c)
  %e = e::eee(%d)
  return (%e))IR",
      &graph);

  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%a)
  %d = d::ddd(%b, %c)
  return (%d))IR",
      &pattern1);
  AT_ASSERT(!findPatternMatches(pattern1, graph).empty());

  // Check that order of nodes inside the diamond does not affect the result
  parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %c = c::ccc(%a)
  %b = b::bbb(%a)
  %d = d::ddd(%b, %c)
  return (%d))IR",
      &pattern2);
  AT_ASSERT(!findPatternMatches(pattern2, graph).empty());
}

/**
 * Test diamond pattern:
 *
 *     i0
 *      |
 *    chunk
 *    /   \
 * os[0] os[1]
 *     \ /
 *      *
 *      |
 *      o1
 */
TEST(SubgraphMatcherTest, Diamond2) {
  Graph graph;
  auto* g_in = graph.addInput();

  auto* g_chunk =
      graph.insertNode(graph.create(prim::ConstantChunk, /*num_outputs =*/2));
  g_chunk->i_(attr::chunks, 2)->i_(attr::dim, 0);
  g_chunk->addInput(g_in);

  auto* g_mul = graph.insertNode(graph.create(aten::mul, /*num_outputs =*/1));
  g_mul->addInput(g_chunk->outputs()[0]);
  g_mul->addInput(g_chunk->outputs()[1]);
  graph.registerOutput(g_mul->output());

  Graph pattern;
  auto* p_in = pattern.addInput();
  auto* p_chunk = pattern.insertNode(
      pattern.create(prim::ConstantChunk, /*num_outputs =*/2));
  p_chunk->i_(attr::chunks, 2)->i_(attr::dim, 0);
  p_chunk->addInput(p_in);

  auto* p_mul =
      pattern.insertNode(pattern.create(aten::mul, /*num_outputs =*/1));
  p_mul->addInput(p_chunk->outputs()[0]);
  p_mul->addInput(p_chunk->outputs()[1]);
  pattern.registerOutput(p_mul->output());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(p_in) == g_in);
    AT_ASSERT(m.values_map.at(p_chunk->outputs()[0]) == g_chunk->outputs()[0]);
    AT_ASSERT(m.values_map.at(p_chunk->outputs()[1]) == g_chunk->outputs()[1]);
    AT_ASSERT(m.values_map.at(p_mul->output()) == g_mul->output());
    AT_ASSERT(m.nodes_map.at(p_mul) == g_mul);
  }
}

TEST(SubgraphMatcherTest, XPattern) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%0, %1):
  %b = b::bbb(%0)
  %c = c::ccc(%1)
  %x = x::xxx(%b, %c)
  %e = e::eee(%x)
  %f = f::fff(%x)
  %g = g::ggg(%e, %f)
  return (%g))IR",
      &graph);
  parseIR(
      R"IR(
graph(%0, %1):
  %b = b::bbb(%0)
  %c = c::ccc(%1)
  %x = x::xxx(%b, %c)
  %e = e::eee(%x)
  %f = f::fff(%x)
  %g = g::ggg(%e, %f)
  return (%g))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

TEST(SubgraphMatcherTest, MultipleMatches) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  return (%t1))IR",
      &pattern);
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 4);
}

TEST(SubgraphMatcherTest, OverlappingMatches) {
  Graph graph, pattern;
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  return (%t2))IR",
      &pattern);
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 3);
}

TEST(SubgraphMatcherTest, MatchInBasicBlocks1) {
  Graph graph;
  parseIR(
      R"IR(
graph(%a, %b, %c):
  %d = aten::mul(%a, %b)
  %x = prim::If(%c)
    block0():
      %x1 = aten::mul(%a, %d)
      -> (%x1)
    block1():
      %x2 = aten::mul(%b, %d)
      -> (%x2)
  return (%x))IR",
      &graph);

  // Ensure the matches don't cross basic block boundaries
  Graph pattern0;
  parseIR(
      R"IR(
graph(%x, %y):
  %z = aten::mul(%x, %y)
  return (%z))IR",
      &pattern0);
  AT_ASSERT(findPatternMatches(pattern0, graph).size() == 3);

  Graph pattern1;
  parseIR(
      R"IR(
graph(%x, %y):
  %z1 = aten::mul(%x, %y)
  %z2 = aten::mul(%y, %z1)
  return (%z2))IR",
      &pattern1);
  AT_ASSERT(findPatternMatches(pattern1, graph).size() == 0);
}

TEST(SubgraphMatcherTest, MatchInBasicBlocks2) {
  Graph graph;
  parseIR(
      R"IR(
graph(%a, %b):
  %x = my::mul(%a, %b)
  %y = my::node_with_subblock()
    block0():
      %z = my::mul(%b, %x)
      -> (%z)
  return (%y))IR",
      &graph);

  // Check that we can match both mul ops
  Graph pattern0;
  parseIR(
      R"IR(
graph(%x, %y):
  %z = my::mul(%x, %y)
  return (%z))IR",
      &pattern0);
  AT_ASSERT(findPatternMatches(pattern0, graph).size() == 2);

  // Ensure the matches don't cross basic block boundaries
  Graph pattern1;
  parseIR(
      R"IR(
graph(%x, %y):
  %u = my::mul(%x, %y)
  %v = my::mul(%y, %u)
  return (%v))IR",
      &pattern1);
  AT_ASSERT(findPatternMatches(pattern1, graph).size() == 0);
}

TEST(SubgraphMatcherTest, MatchesAttributes) {
  Graph graph;
  parseIR(
      R"IR(
graph(%0):
  %a = a::a[isattr=[1,2]](%0)
  %b = a::b[intattr=10, floatattr=3.14, complexattr=-3.14j](%0)
  %c = a::c[myattr="qqq"](%a, %b)
  return (%c))IR",
      &graph);

  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%a, %b):
  %c = a::c[myattr="qqq"](%a, %b)
  return (%c))IR",
        &pattern);
    AT_ASSERT(!findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%a, %b):
  %c = a::c[myattr="zzz"](%a, %b)
  return (%c))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%0):
  %b = a::b[extraattr=10](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%0):
  %b = a::b[intattr=10, floatattr=3.14, complexattr=-3.14j](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(!findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%0):
  %b = a::b[intattr=10, floatattr=3.14, complexattr=-3.14j, strattr="rrr"](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%0):
  %a = a::a[isattr=[1,2]](%0)
  return (%a))IR",
        &pattern);
    // Lists are not supported yet, thus we shouldn't match for now.
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    parseIR(
        R"IR(
graph(%a, %b):
  %c = a::c[myattr="q.*"](%a, %b)
  return (%c))IR",
        &pattern);
    AT_ASSERT(!findPatternMatches(pattern, graph).empty());
  }
}

TEST(SubgraphMatcherTest, BadPattern) {
  Graph graph, pattern1, pattern2;
  parseIR(
      R"IR(
graph(%x):
  %y = my::op1(%x)
  %z = my::op2(%x)
  return (%y, %z))IR",
      &graph);

  parseIR(
      R"IR(
graph(%x):
  %y = my::node_with_subblock()
    block0():
      %z = my::op(%x)
      -> (%z)
  return (%y))IR",
      &pattern1);
  // No support for patterns with subblocks
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(findPatternMatches(pattern1, graph));

  parseIR(
      R"IR(
graph(%x):
  %y = my::op1(%x)
  %z = my::op2(%x)
  return (%y, %z))IR",
      &pattern2);
  // Not supported multi-output pattern, because not the whole pattern is
  // covered by a traversal up from the first output (`%z = ...` is not
  // visited). See the note "Multi-output Patterns" in subgraph_matcher.h.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(findPatternMatches(pattern2, graph));
}

TEST(SubgraphMatcherTest, MultiOutput) {
  {
    Graph graph, pattern;
    parseIR(
        R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%a, %b)
  %x = a::aaa(%c)
  %y = b::bbb(%x)
  %z = d::ddd(%x, %y)
  return (%y))IR",
        &graph);
    parseIR(
        R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  return (%b, %a))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).size() == 2);
  }
  {
    Graph graph, pattern;
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
        &graph);
    parseIR(
        R"IR(
graph(%0, %1):
  %a1, %a2 = a::aaa(%0, %1)
  %b = b::bbb(%a1)
  return (%b, %a2))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).size() == 2);
  }
}

} // namespace jit
} // namespace torch
