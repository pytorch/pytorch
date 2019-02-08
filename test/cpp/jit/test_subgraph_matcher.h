#pragma once

#include "test/cpp/jit/test_base.h"

#include "torch/csrc/jit/passes/utils/subgraph_matcher.h"

#include <random>

namespace torch {
namespace jit {
namespace {

void testTrivial() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  auto o0 = i0.tanh();

  Graph mg;

  Var mi0 = Var::asNewInput(mg);
  auto mo0 = mi0.tanh();

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[mi0.value()] == i0.value());
    AT_ASSERT(m[mo0.value()] == o0.value());
    AT_ASSERT(m[mo0.value()->node()] == o0.value()->node());
  }
  AT_ASSERT(hit == 1);
}

void testLinear() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  auto o0 = i0.tanh();
  auto o1 = o0.tanh();

  Graph mg;
  auto m0 = Var::asNewInput(mg);
  auto m1 = m0.tanh();
  auto m2 = m1.tanh();

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[m0.value()] == i0.value());
    AT_ASSERT(m[m1.value()] == o0.value());
    AT_ASSERT(m[m1.value()->node()] == o0.value()->node());
    AT_ASSERT(m[m2.value()] == o1.value());
    AT_ASSERT(m[m2.value()->node()] == o1.value()->node());
  }
  AT_ASSERT(hit == 1);
}

void testMultipleInput() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  Var i1 = Var::asNewInput(graph);
  auto o0 = i0 * i1;

  Graph mg;
  auto m0 = Var::asNewInput(mg);
  auto m1 = Var::asNewInput(mg);
  auto m2 = m0 * m1;

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[m0.value()] == i0.value());
    AT_ASSERT(m[m1.value()] == i1.value());
    AT_ASSERT(m[m2.value()] == o0.value());
    AT_ASSERT(m[m2.value()->node()] == o0.value()->node());
  }
  AT_ASSERT(hit == 1);
}

void testMultipleOutput() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  auto os = i0.chunk(2, 0);
  auto o0 = os[0];
  auto o1 = os[1];

  Graph mg;
  auto mi0 = Var::asNewInput(mg);
  auto ms = mi0.chunk(2, 0);
  auto mo0 = ms[0];
  auto mo1 = ms[1];

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[mi0.value()] == i0.value());
    AT_ASSERT(m[mo0.value()] == o0.value());
    AT_ASSERT(m[mo1.value()] == o1.value());
    AT_ASSERT(m[mo0.value()->node()] == o0.value()->node());
    AT_ASSERT(m[mo1.value()->node()] == o1.value()->node());
  }
  AT_ASSERT(hit == 1);
}

void testMultipleMatches() {
  Graph graph;

  auto match_count = 3;
  Var i0 = Var::asNewInput(graph);
  auto os0 = i0.chunk(2, 0);
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 1);

  for (auto i = 0; i < match_count - 1; ++i) {
    os0 = os0[distribution(generator)].chunk(2, 0);
  }

  Graph mg;
  auto mi0 = Var::asNewInput(mg);
  auto ms = mi0.chunk(2, 0);
  auto mo0 = ms[0];
  auto mo1 = ms[1];

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
  }
  AT_ASSERT(hit == match_count);
}

// Test diamond pattern:
//
//     i0
//      |
//    chunk
//    /   \
// os[0] os[1]
//     \ /
//      *
//      |
//      o1
//
void testDiamond() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  auto os = i0.chunk(2, 0);
  auto o1 = os[0] * os[1];

  Graph mg;
  auto mi0 = Var::asNewInput(mg);
  auto ms = mi0.chunk(2, 0);
  auto mo = ms[0] * ms[1];

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[mi0.value()] == i0.value());
    AT_ASSERT(m[ms[0].value()] == os[0].value());
    AT_ASSERT(m[ms[1].value()] == os[1].value());
    AT_ASSERT(m[mo.value()->node()] == o1.value()->node());
    AT_ASSERT(m[mo.value()] == o1.value());
  }
  AT_ASSERT(hit == 1);
}

// Test X pattern:
//
//   i0   i1
//     \ /
//      *
//     o0
//      |
//    chunk
//    /   \
// os[0] os[1]
//
void testX() {
  Graph graph;

  Var i0 = Var::asNewInput(graph);
  Var i1 = Var::asNewInput(graph);
  auto o0 = i0 * i1;
  auto os = o0.chunk(2, 0);

  Graph mg;
  auto mi0 = Var::asNewInput(mg);
  auto mi1 = Var::asNewInput(mg);
  auto mo0 = mi0 * mi1;
  auto ms = mo0.chunk(2, 0);

  int hit = 0;
  for (const auto& m : MatchIterator(mg, graph)) {
    hit++;
    AT_ASSERT(m[mi0.value()] == i0.value());
    AT_ASSERT(m[mi1.value()] == i1.value());
    AT_ASSERT(m[mo0.value()->node()] == o0.value()->node());
    AT_ASSERT(m[mo0.value()] == o0.value());
    AT_ASSERT(m[ms[0].value()->node()] == os[0].value()->node());
    AT_ASSERT(m[ms[0].value()] == os[0].value());
    AT_ASSERT(m[ms[1].value()] == os[1].value());
  }
  AT_ASSERT(hit == 1);
}

void testMatchInBasicBlocks() {
  Graph graph;

  Var a = Var::asNewInput(graph);
  Var b = Var::asNewInput(graph);
  Var c = a * b;
  auto r = graph.appendNode(
      graph.create(prim::If, {Var::asNewInput(graph).value()}));
  auto then_block = r->addBlock();
  auto else_block = r->addBlock();
  {
    WithInsertPoint guard(then_block);
    auto d = b * c;
    then_block->registerOutput(d.value());
  }
  {
    WithInsertPoint guard(else_block);
    auto d = a * c;
    else_block->registerOutput(d.value());
  }

  // Ensure the matches don't cross basic block boundaries
  Graph mg0;
  auto m0 = Var::asNewInput(mg0);
  auto m1 = Var::asNewInput(mg0);
  auto m2 = m0 * m1;

  int hit = 0;
  for (const auto& m : MatchIterator(mg0, graph)) {
    hit++;
    AT_ASSERT(m[m0.value()] == a.value());
    AT_ASSERT(m[m1.value()] == b.value());
    AT_ASSERT(m[m2.value()] == c.value());
    AT_ASSERT(m[m2.value()->node()] == c.value()->node());
  }
  AT_ASSERT(hit == 1);

  Graph mg1;
  m0 = Var::asNewInput(mg1);
  m1 = Var::asNewInput(mg1);
  m2 = m0 * m1;
  auto m3 = m1 * m2;

  hit = 0;
  for (const auto& m : MatchIterator(mg1, graph)) {
    hit++;
  }
  AT_ASSERT(hit == 0);
}

void testSubgraphMatching() {
  testTrivial();
  testLinear();
  testMultipleInput();
  testMultipleOutput();
  testMultipleMatches();
  testDiamond();
  testX();
  testMatchInBasicBlocks();
}

} // namespace
} // namespace jit
} // namespace torch
