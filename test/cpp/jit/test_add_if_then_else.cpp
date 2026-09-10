#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/add_if_then_else.h>

namespace torch {
namespace jit {

TEST(AddIfThenElseOpTest, AddIfThenElseOpSimple) {
  const auto src = R"IR(
        graph(%cond: bool, %a: Tensor, %b: Tensor):
            %result: Tensor = prim::If(%cond)
                block0():
                    -> (%a)
                block1():
                    -> (%b)
            return (%result)
    )IR";

  auto graph = std::make_shared<Graph>();
  parseIR(src, graph.get());
  EXPECT_TRUE(AddIfThenElseOp(graph));

  testing::FileCheck()
      .check_count("= prim::IfThenElse", 1, /*exactly*/ true)
      ->check_count("= prim::If", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(AddIfThenElseOpTest, NoIfThenElseOpMultipleOutputs) {
  const auto src = R"IR(
        graph(%cond: bool, %a: Tensor, %b: Tensor):
            %result1: Tensor, %result2: Tensor = prim::If(%cond)
                block0():
                    -> (%a, %b)
                block1():
                    -> (%b, %a)
            return (%result1, %result2)
    )IR";

  auto graph = std::make_shared<Graph>();
  parseIR(src, graph.get());
  EXPECT_FALSE(AddIfThenElseOp(graph));

  testing::FileCheck()
      .check_count("= prim::IfThenElse", 0, /*exactly*/ true)
      ->check_count("= prim::If", 1, /*exactly*/ true)
      ->run(*graph);
}

} // namespace jit
} // namespace torch
