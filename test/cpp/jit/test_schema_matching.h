#pragma once

#include <torch/jit.h>
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

void testSchemaMatching() {
  auto run = [](std::shared_ptr<script::CompilationUnit> cu, const std::string& name, std::vector<IValue> stack) {
    auto graph = cu->get_function(name).graph();
    Code code(graph);
    InterpreterState interp(code);
    interp.run(stack);
    return stack;
  };
  {
    RegisterOperators reg({
        Operator(
            "aten::test_vartype(t[] a, t b) -> (t)",
            [](const Node* node) {
              return [](Stack& stack) {
                c10::List<double> list;
                double a;
                pop(stack, list, a);
                push(stack, a);
                return 0;
              };
            }),
    });
    static const auto cf_examples = R"JIT(
      def test():
          a = (1.0, 2.0)
          return torch.test_vartype(a, 2.0)
    )JIT";
    auto cu = compile(cf_examples);
    auto graph = cu->get_function("test").graph();
    graph->dump();
    std::vector<IValue> stack;
    run(cu, "test", stack);
    TORCH_INTERNAL_ASSERT(pop(stack).toDouble() == 2.0);
  }
  {
    RegisterOperators reg({
        Operator(
            "aten::test_vartype2(t a, t[] b) -> (t)",
            [](const Node* node) {
              return [](Stack& stack) {
                double a;
                c10::List<double> list;
                pop(stack, a, list);
                push(stack, a);
                return 0;
              };
            }),
    });
    static const auto cf_examples = R"JIT(
      def test():
          a = (1.0, 2.0)
          return torch.test_vartype2(2.0, a)
    )JIT";
    auto cu = compile(cf_examples);
    auto graph = cu->get_function("test").graph();
    graph->dump();


    static const auto cf_example_fail = R"JIT(
      def test():
          a = (1, 2)
          return torch.test_vartype2(2.0, a)
    )JIT";
    auto cu2 = compile(cf_example_fail);
  }



}
} // namespace jit
} // namespace torch
