#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/subgraph_matcher.h"

namespace torch {
namespace jit {

void testTrivial1() {
  Graph graph, pattern;
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  return (%a))IR",
      &graph);
  script::parseIR(
      R"IR(
graph(%0):
  %x = a::aaa(%0)
  return (%x))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

void testTrivial2() {
  Graph graph;
  Var i0 = Var::asNewInput(graph);
  auto o0 = i0.tanh();
  graph.registerOutput(o0.value());

  Graph pattern;
  Var mi0 = Var::asNewInput(pattern);
  auto mo0 = mi0.tanh();
  pattern.registerOutput(mo0.value());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(mi0.value()) == i0.value());
    AT_ASSERT(m.values_map.at(mo0.value()) == o0.value());
    AT_ASSERT(m.nodes_map.at(mo0.value()->node()) == o0.value()->node());
  }
}

void testTrivial3() {
  Graph graph, pattern;
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::a(%0)
  %b = a::b(%0)
  %c = a::c(%a, %b)
  return (%c))IR",
      &graph);
  script::parseIR(
      R"IR(
graph(%a, %b):
  %c = a::c(%a, %b)
  return (%c))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

void testTrivial4() {
  Graph graph;
  Var i0 = Var::asNewInput(graph);
  Var i1 = Var::asNewInput(graph);
  auto o0 = i0 * i1;
  graph.registerOutput(o0.value());

  Graph pattern;
  auto m0 = Var::asNewInput(pattern);
  auto m1 = Var::asNewInput(pattern);
  auto m2 = m0 * m1;
  pattern.registerOutput(m2.value());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(m0.value()) == i0.value());
    AT_ASSERT(m.values_map.at(m1.value()) == i1.value());
    AT_ASSERT(m.values_map.at(m2.value()) == o0.value());
    AT_ASSERT(m.nodes_map.at(m2.value()->node()) == o0.value()->node());
  }
}

void testLinear1() {
  Graph graph, pattern;
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  %b = b::bbb(%a)
  %c = c::ccc(%b)
  %d = d::ddd(%c)
  %a = a::aaa(%0)
  return (%d))IR",
      &graph);
  script::parseIR(
      R"IR(
graph(%0):
  %x = b::bbb(%0)
  %y = c::ccc(%x)
  return (%y))IR",
      &pattern);
  AT_ASSERT(!findPatternMatches(pattern, graph).empty());
}

void testLinear2() {
  Graph graph;
  Var i0 = Var::asNewInput(graph);
  auto o0 = i0.tanh();
  auto o1 = o0.tanh();
  graph.registerOutput(o1.value());

  Graph pattern;
  auto m0 = Var::asNewInput(pattern);
  auto m1 = m0.tanh();
  auto m2 = m1.tanh();
  pattern.registerOutput(m2.value());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(m0.value()) == i0.value());
    AT_ASSERT(m.values_map.at(m1.value()) == o0.value());
    AT_ASSERT(m.values_map.at(m2.value()) == o1.value());
    AT_ASSERT(m.nodes_map.at(m1.value()->node()) == o0.value()->node());
    AT_ASSERT(m.nodes_map.at(m2.value()->node()) == o1.value()->node());
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
void testDiamond1() {
  Graph graph, pattern1, pattern2;
  script::parseIR(
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

  script::parseIR(
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
  script::parseIR(
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
void testDiamond2() {
  Graph graph;
  Var i0 = Var::asNewInput(graph);
  auto os = i0.chunk(2, 0);
  auto o1 = os[0] * os[1];

  Graph pattern;
  auto mi0 = Var::asNewInput(pattern);
  auto ms = mi0.chunk(2, 0);
  auto mo = ms[0] * ms[1];
  pattern.registerOutput(mo.value());

  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 1);
  for (const Match& m : matches) {
    AT_ASSERT(m.values_map.at(mi0.value()) == i0.value());
    AT_ASSERT(m.values_map.at(ms[0].value()) == os[0].value());
    AT_ASSERT(m.values_map.at(ms[1].value()) == os[1].value());
    AT_ASSERT(m.values_map.at(mo.value()) == o1.value());
    AT_ASSERT(m.nodes_map.at(mo.value()->node()) == o1.value()->node());
  }
}

void testXPattern() {
  Graph graph, pattern;
  script::parseIR(
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
  script::parseIR(
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

void testMultipleMatches() {
  Graph graph, pattern;
  script::parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  script::parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  return (%t1))IR",
      &pattern);
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 4);
}

void testOverlappingMatches() {
  Graph graph, pattern;
  script::parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  %t3 = a::aaa(%t2)
  %t4 = a::aaa(%t3)
  return (%t4))IR",
      &graph);
  script::parseIR(
      R"IR(
graph(%t0):
  %t1 = a::aaa(%t0)
  %t2 = a::aaa(%t1)
  return (%t2))IR",
      &pattern);
  auto matches = findPatternMatches(pattern, graph);
  AT_ASSERT(matches.size() == 3);
}

void testMatchInBasicBlocks1() {
  Graph graph;
  script::parseIR(
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
  script::parseIR(
      R"IR(
graph(%x, %y):
  %z = aten::mul(%x, %y)
  return (%z))IR",
      &pattern0);
  AT_ASSERT(findPatternMatches(pattern0, graph).size() == 3);

  Graph pattern1;
  script::parseIR(
      R"IR(
graph(%x, %y):
  %z1 = aten::mul(%x, %y)
  %z2 = aten::mul(%y, %z1)
  return (%z2))IR",
      &pattern1);
  AT_ASSERT(findPatternMatches(pattern1, graph).size() == 0);
}

void testMatchInBasicBlocks2() {
  Graph graph;
  script::parseIR(
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
  script::parseIR(
      R"IR(
graph(%x, %y):
  %z = my::mul(%x, %y)
  return (%z))IR",
      &pattern0);
  AT_ASSERT(findPatternMatches(pattern0, graph).size() == 2);

  // Ensure the matches don't cross basic block boundaries
  Graph pattern1;
  script::parseIR(
      R"IR(
graph(%x, %y):
  %u = my::mul(%x, %y)
  %v = my::mul(%y, %u)
  return (%v))IR",
      &pattern1);
  AT_ASSERT(findPatternMatches(pattern1, graph).size() == 0);
}

void testMatchesAttributes() {
  Graph graph;
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::a[isattr=[1,2]](%0)
  %b = a::b[intattr=10, floatattr=3.14](%0)
  %c = a::c[myattr="qqq"](%a, %b)
  return (%c))IR",
      &graph);

  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%a, %b):
  %c = a::c[myattr="qqq"](%a, %b)
  return (%c))IR",
        &pattern);
    AT_ASSERT(!findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%a, %b):
  %c = a::c[myattr="zzz"](%a, %b)
  return (%c))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%0):
  %b = a::b[extraattr=10](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%0):
  %b = a::b[intattr=10, floatattr=3.14](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(!findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%0):
  %b = a::b[intattr=10, floatattr=3.14, strattr="rrr"](%0)
  return (%b))IR",
        &pattern);
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
  {
    Graph pattern;
    script::parseIR(
        R"IR(
graph(%0):
  %a = a::a[isattr=[1,2]](%0)
  return (%a))IR",
        &pattern);
    // Lists are not supported yet, thus we shouldn't match for now.
    AT_ASSERT(findPatternMatches(pattern, graph).empty());
  }
}

void testBadPattern() {
  Graph graph, pattern1, pattern2;
  script::parseIR(
      R"IR(
graph(%0):
  %a = a::aaa(%0)
  return (%a))IR",
      &graph);

  script::parseIR(
      R"IR(
graph(%x):
  %y = my::node_with_subblock()
    block0():
      %z = my::op(%x)
      -> (%z)
  return (%y))IR",
      &pattern1);
  ASSERT_ANY_THROW(findPatternMatches(pattern1, graph));

  script::parseIR(
      R"IR(
graph(%x):
  %y = my::op1(%x)
  %z = my::op2(%x)
  return (%y, %z))IR",
      &pattern2);
  ASSERT_ANY_THROW(findPatternMatches(pattern2, graph));
}

void testSubgraphMatching() {
  testTrivial1();
  testTrivial2();
  testTrivial3();
  testTrivial4();
  testLinear1();
  testLinear2();
  testDiamond1();
  testDiamond2();
  testXPattern();
  testMultipleMatches();
  testOverlappingMatches();
  testMatchInBasicBlocks1();
  testMatchInBasicBlocks2();
  testMatchesAttributes();
  testBadPattern();
}

} // namespace jit
} // namespace torch
