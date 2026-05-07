#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

TEST(ConstantPoolingTest, Int) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %8 : int = prim::Constant[value=1]()
  %10 : int = prim::Constant[value=1]()
  return (%8, %10)
  )IR",
      &*graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count("prim::Constant", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, PoolingAcrossBlocks) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%cond : Tensor):
  %a : str = prim::Constant[value="bcd"]()
  %3 : bool = aten::Bool(%cond)
  %b : str = prim::If(%3)
    block0():
      %b.1 : str = prim::Constant[value="abc"]()
      -> (%b.1)
    block1():
      %b.2 : str = prim::Constant[value="abc"]()
      -> (%b.2)
  %7 : (str, str) = prim::TupleConstruct(%a, %b)
  return (%7)
  )IR",
      &*graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count("prim::Constant[value=\"abc\"]", 1, /*exactly*/ true)
      ->check_count("prim::Constant[value=\"bcd\"]", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, PoolingDifferentDevices) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %2 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=1]()
  %5 : int? = prim::Constant()
  %7 : Device? = prim::Constant()
  %15: bool = prim::Constant[value=0]()
  %10 : int = prim::Constant[value=6]()
  %3 : int[] = prim::ListConstruct(%1, %2)
  %x : Tensor = aten::tensor(%3, %5, %7, %15)
  %y : Tensor = aten::tensor(%3, %10, %7, %15)
  %9 : int[] = prim::ListConstruct(%1, %2)
  %z : Tensor = aten::tensor(%9, %10, %7, %15)
  prim::Print(%x, %y, %z)
  return (%1)
  )IR",
      &*graph);
  // three tensors created - two different devices among the three
  // don't have good support for parsing tensor constants
  ConstantPropagation(graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count(
          "Float(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->check_count(
          "Long(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, DictConstantPooling) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %0 : int = prim::Constant[value=1]() # test/elias.py:6:9
  %1 : int = prim::Constant[value=2]() # test/elias.py:6:12
  %a.1 : Dict(int, int) = prim::DictConstruct(%0, %1)
  %b.1 : Dict(int, int) = prim::DictConstruct(%1, %1)
  return (%a.1, %b.1)
  )IR",
      &*graph);
  ConstantPropagation(graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count(
          "Dict(int, int) = prim::Constant",
          2,
          /*exactly*/ true)
      ->run(*graph);
}
} // namespace jit
} // namespace torch
