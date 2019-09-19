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
  {
    RegisterOperators reg({
        Operator(
            "prim::test_tuple() -> (float[])",
            [](const Node* node) {
              return [](Stack& stack) {
                c10::List<double> list;
                auto li = IValue(list);
                std::vector<IValue> tup = {li};
                push(stack, c10::ivalue::Tuple::create(tup));
                return 0;
              };
            },
            _aliasAnalysisFromSchema()),
        Operator(
            "prim::run_float_list(float[] a) -> (int)",
            [](const Node* node) {
              return [](Stack& stack) {
                pop(stack);
                push(stack, 1);
                return 0;
              };
            },
            _aliasAnalysisFromSchema()),
    });
    auto graph = std::make_shared<Graph>();
    script::parseIR(
        R"IR(
  graph():
    %2 : (float[]) = prim::test_tuple()
    %1 : int = prim::Constant[value=0]()
    %y : float[] = prim::TupleIndex(%2, %1)
    %z : int = prim::run_float_list(%y)
    return (%z)
    )IR",
        &*graph);
    ConstantPropagation(graph);
    // float[] are not embeddable as constants, so we should not
    // run the run_float_list op.
    // this logic prevents e.g. running a tensor with grad in constant prop
    testing::FileCheck()
        .check("test_tuple")
        ->check("TupleIndex")
        ->check("run_float_list")
        ->run(*graph);
  }
}
} // namespace jit
} // namespace torch
