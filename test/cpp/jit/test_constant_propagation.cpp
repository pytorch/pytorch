#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>
#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/custom_operator.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

namespace {
inline c10::OperatorOptions _aliasAnalysisFromSchema() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}
} // namespace

void testConstantPropagation() {
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %1 : int = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=0]()
  %x : (int, int) = prim::TupleConstruct(%0, %1)
  %y : int = prim::TupleIndex(%x, %0)
  %5 : int = aten::add(%y, %y)
  return (%5)
  )IR",
        &*graph);
    // optimize through tuple construct and indexing
    ConstantPropagation(graph);
    testing::FileCheck()
        .check("graph")
        ->check_next("prim::Constant[value=0]")
        ->check_next("return")
        ->run(*graph);
  }
  {
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
graph():
  %10 : None = prim::Constant()
  %7 : int = prim::Constant[value=0]()
  %1 : int = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=3]()
  %x : (int, int) = prim::TupleConstruct(%0, %1)
  %y : (int, (int, int)) = prim::TupleConstruct(%1, %x)
  %6 : (int, int) = prim::TupleIndex(%y, %1)
  %z : int = prim::TupleIndex(%6, %7)
  %9 : int = aten::add(%z, %z)
  %ign = prim::Print(%y, %9)
  return (%10)  )IR",
        &*graph);
    ConstantPropagation(graph);
    // The index should be optimized away, with a computed value of 6,
    // and the TupleConstructs should still remain
    testing::FileCheck()
        .check_count("TupleConstruct", 2)
        ->check_not("TupleIndex")
        ->check("value=6")
        ->run(*graph);
  }
}
} // namespace jit
} // namespace torch
