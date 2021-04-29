#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/peephole.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, IsAndIsNot)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, IsAndIsNot2) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, IsAndIsNot3) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, UnwrapOptional)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, UnwrapOptional2) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(PeepholeOptimizeTest, AddMMFusion) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
      graph(
        %0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(1, 1, 1)):
        %3 : int = prim::Constant[value=1]()
        %4 : Tensor = aten::mm(%0, %1)
        %5 : Tensor = aten::add(%4, %2, %3)
        %6 : Tensor = aten::add(%5, %2, %3)
        return (%6)
        )IR",
      graph.get());
  FuseAddMM(graph);
  testing::FileCheck().check("addmm")->run(*graph);
}
} // namespace jit
} // namespace torch
