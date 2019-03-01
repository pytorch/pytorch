#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include <torch/types.h>
#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/graph_executor.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

namespace {

using FileCheck = testing::FileCheck;

void checkIntResultEqualForInput(
    std::shared_ptr<Graph> g1,
    std::shared_ptr<Graph> g2,
    Stack s1) {
  Stack s2 = s1;
  GraphExecutor e1(g1);
  e1.run(s1);

  GraphExecutor e2(g1);
  e1.run(s2);

  AT_ASSERT(s1[0].toInt() == s2[0].toInt());
}

std::string to_str(std::shared_ptr<Graph> g) {
  std::stringstream g_str;
  g_str << *g;
  return g_str.str();
}
} // namespace
static constexpr int64_t kUnrollFactor = 8;

void test_loop_unrolling() {
  auto graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
graph(%x : Tensor):
  %3 : bool = prim::Constant[value=1]()
  %y.1 : int = prim::Constant[value=0]()
  %2 : int = prim::Int(%x)
  %y : int = prim::Loop(%2, %3, %y.1)
    block0(%i : int, %5 : int):
      %y.2 : int = aten::sub(%5, %i)
      -> (%3, %y.2)
  return (%y))IR",
      &*graph);

  auto orig_graph = graph->copy();
  UnrollLoops(graph);
  Stack s1;
  auto input1 = torch::tensor(10);
  s1.push_back(input1);

  checkIntResultEqualForInput(orig_graph, graph, s1);
  // check 8 subs then loop with another sub
  FileCheck()
      .check("prim::Loop")
      ->check_count("aten::sub", kUnrollFactor)
      ->check("prim::Loop")
      ->check("aten::sub")
      ->run(to_str(graph));
}

void test_loop_unrolling_const_increment() {
  // test loop unrolling const decrement
  auto graph = std::make_shared<Graph>();
  // y decremented by 1 on each loop
  script::parseIR(
      R"IR(
  graph():
    %2 : bool = prim::Constant[value=1]()
    %y.1 : int = prim::Constant[value=0]()
    %1 : int = prim::Constant[value=10]()
    %5 : int = prim::Constant[value=1]()
    %y : int = prim::Loop(%1, %2, %y.1)
      block0(%3 : int, %4 : int):
        %y.2 : int = aten::sub(%4, %5)
        -> (%2, %y.2)
    return (%y))IR",
      &*graph);

  auto orig_graph = graph->copy();
  UnrollLoops(graph);
  std::stringstream g_str;
  g_str << *graph;
  FileCheck().check_not("prim::Loop")->run(to_str(graph));

  checkIntResultEqualForInput(orig_graph, graph, {});
}

void test_loop_unrolling_iter_increment() {
  // test loop unrolling decrement by iter value for constant range
  auto graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
  graph():
    %2 : bool = prim::Constant[value=1]()
    %y.1 : int = prim::Constant[value=0]()
    %1 : int = prim::Constant[value=10]()
    %y : int = prim::Loop(%1, %2, %y.1)
      block0(%i : int, %4 : int):
        %y.2 : int = aten::add(%4, %i)
        -> (%2, %y.2)
    return (%y)
  )IR",
      &*graph);

  auto orig_graph = graph->copy();
  UnrollLoops(graph);
  std::stringstream g_str;
  g_str << *graph;
  FileCheck().check_not("prim::Loop")->run(to_str(graph));

  checkIntResultEqualForInput(orig_graph, graph, {});
}

void test_loop_unrolling_nested_loops() {
  // test nested loop
  auto graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
  graph(%x : Tensor):
    %3 : bool = prim::Constant[value=1]()
    %y.1 : int = prim::Constant[value=0]()
    %2 : int = prim::Constant[value=10]()
    %y : int = prim::Loop(%2, %3, %y.1)
      block0(%4 : int, %9 : int):
        %6 : int = prim::Int(%x)
        %y.3 : int = prim::Loop(%6, %3, %9)
          block0(%j : int, %10 : int):
            %y.2 : int = aten::sub(%10, %j)
            -> (%3, %y.2)
        -> (%3, %y.3)
    return (%y)
    )IR",
      &*graph);

  auto orig_graph = graph->copy();
  UnrollLoops(graph);
  std::stringstream g_str;
  g_str << *graph;
  // inner loop with 8 subs followed by loop epilogue
  FileCheck()
      .check("prim::Loop")
      ->check("prim::Loop")
      ->check_count("aten::sub", kUnrollFactor)
      ->check("prim::Loop")
      ->check("aten::sub")
      ->run(to_str(graph));
  Stack s1;
  s1.push_back(torch::tensor(10));

  checkIntResultEqualForInput(orig_graph, graph, s1);
}

void test_loop_unrolling_nested_unused_counter() {
  auto graph = std::make_shared<Graph>();
  script::parseIR(
      R"IR(
  graph(%x : Tensor):
    %3 : bool = prim::Constant[value=1]()
    %y.1 : int = prim::Constant[value=0]()
    %6 : int = prim::Constant[value=1]()
    %2 : int = prim::Int(%x)
    %y : int = prim::Loop(%2, %3, %y.1)
      block0(%4 : int, %5 : int):
        %y.2 : int = aten::sub(%5, %6)
        -> (%3, %y.2)
    return (%y)
  )IR",
      &*graph);

  auto orig_graph = graph->copy();
  UnrollLoops(graph);

  // unrolled loop should not increment the counter since it is not used
  FileCheck()
      .check("prim::Loop")
      ->check_not("aten::add")
      ->check("return")
      ->run(to_str(graph));

  auto test_inputs = {
      torch::tensor(-20),
      torch::tensor(-2),
      torch::tensor(-1),
      torch::tensor(0),
      torch::tensor(1),
      torch::tensor(2),
  };

  for (auto input : test_inputs) {
    checkIntResultEqualForInput(orig_graph, graph, {input});
  }
}

void testLoopUnrolling(std::ostream& out = std::cout) {
  test_loop_unrolling();
  test_loop_unrolling_const_increment();
  test_loop_unrolling_iter_increment();
  test_loop_unrolling_nested_loops();
  test_loop_unrolling_nested_unused_counter();
}
} // namespace jit
} // namespace torch
