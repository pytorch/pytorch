#pragma once

#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/peephole.h>

namespace torch {
namespace jit {

using namespace script;
using namespace testing;

namespace test {

void testPeepholeOptimize() {
  // test is / is not none optimization
  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph(%0 : int):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
        graph.get());
    PeepholeOptimize(graph);
    testing::FileCheck()
        .check_not("aten::__is__")
        ->check_not("aten::__isnot__")
        ->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph(%0: int?):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
        graph.get());
    PeepholeOptimize(graph);
    testing::FileCheck()
        .check("aten::__is__")
        ->check("aten::__isnot__")
        ->run(*graph);
  }

  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph(%0: int?):
  %1 : Tensor = prim::AutogradZero()
  %2 : None = prim::Constant()
  %4 : bool = aten::__is__(%0, %1)
  %5 : bool = aten::__isnot__(%1, %2)
  return (%4, %5)
  )IR",
        graph.get());
    PeepholeOptimize(graph);
    testing::FileCheck()
        .check("aten::__is__")
        ->check_not("aten::__isnot__")
        ->run(*graph);
  }

  // test unwrap optional
  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph():
  %1 : Float(*, *, *) = prim::Constant()
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
        graph.get());
    PeepholeOptimize(graph);
    testing::FileCheck().check_not("unwrap")->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    parseIR(
        R"IR(
graph(%1 : Float(*, *, *)?):
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
        graph.get());
    PeepholeOptimize(graph);
    testing::FileCheck().check_count("unwrap", 2)->run(*graph);
  }
}
} // namespace test
} // namespace jit
} // namespace torch
