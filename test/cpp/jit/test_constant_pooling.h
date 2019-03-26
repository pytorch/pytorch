#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "test/cpp/jit/test_base.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

void testConstantPooling() {
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
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
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph(%cond : Tensor):
  %a : string = prim::Constant[value="bcd"]()
  %3 : bool = prim::Bool(%cond)
  %b : string = prim::If(%3)
    block0():
      %b.1 : string = prim::Constant[value="abc"]()
      -> (%b.1)
    block1():
      %b.2 : string = prim::Constant[value="abc"]()
      -> (%b.2)
  %7 : (string, string) = prim::TupleConstruct(%a, %b)
  return (%7)
  )IR",
        &*graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("prim::Constant[value=\"abc\"]", 1, /*exactly*/ true)
        ->check_count("prim::Constant[value=\"bcd\"]", 1, /*exactly*/ true)
        ->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %2 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=1]()
  %5 : int? = prim::Constant()
  %7 : Device? = prim::Constant()
  %10 : int = prim::Constant[value=6]()
  %3 : int[] = prim::ListConstruct(%1, %2)
  %x : Tensor = aten::tensor(%3, %5, %7)
  %y : Tensor = aten::tensor(%3, %10, %7)
  %9 : int[] = prim::ListConstruct(%1, %2)
  %z : Tensor = aten::tensor(%9, %10, %7)
  %14 : (Tensor, Tensor) = prim::TupleConstruct(%x, %y)
  return (%14)
  )IR",
        &*graph);
    // three tensors created - two different devices among the three
    // don't have good support for parsing tensor constants
    ConstantPropagation(graph);
    ConstantPooling(graph);
    testing::FileCheck()
        .check_count("Float(2) = prim::Constant", 1, /*exactly*/ true)
        ->check_count("Long(2) = prim::Constant", 1, /*exactly*/ true)
        ->run(*graph);
  }
}

} // namespace jit
} // namespace torch
