#pragma once

#include <ATen/test/test_assert.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include "test/cpp/jit/test_base.h"
#include "torch/csrc/jit/custom_operator.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

void testSchemaMatching() {
  auto run = [](std::shared_ptr<script::CompilationUnit> cu,
                const std::string& name,
                std::vector<IValue>& stack) {
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
    static const auto exam1 = R"JIT(
      def test():
          a = (1.0, 2.0)
          return torch.test_vartype(a, 2.0)
    )JIT";
    auto cu = compile(exam1);
    auto graph = cu->get_function("test").graph();
    std::vector<IValue> stack;
    run(cu, "test", stack);
    TORCH_INTERNAL_ASSERT(pop(stack).toDouble() == 2.0);

    static const auto error_example = R"JIT(
      def test():
          a = (1.0, 2.0)
          non_float = (1, 1)
          return torch.test_vartype(a, non_float)
    )JIT";

    ASSERT_THROWSM(compile(error_example), "previously matched to type");
  }
  {
    RegisterOperators reg({
        Operator(
            "aten::test_vartype2(t a, t[] b) -> (t[])",
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
    static const auto exam2 = R"JIT(
      def test():
          a = (1.0, 2.0)
          return torch.test_vartype2(3.0, a)
    )JIT";
    auto cu = compile(exam2);
    std::vector<IValue> stack;
    run(cu, "test", stack);
    TORCH_INTERNAL_ASSERT(pop(stack).toDouble() == 3.0);

    static const auto error_exam2 = R"JIT(
      def test():
          a = (1, 2)
          return torch.test_vartype2(3.0, a)
    )JIT";
    ASSERT_THROWSM(compile(error_exam2), "previously matched to type");
  }
}
} // namespace jit
} // namespace torch
