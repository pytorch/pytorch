#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/add_ternary_op.h>

namespace torch {
namespace jit {

TEST(AddTernaryOpTest, AddTernaryOpSimple) {
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
  EXPECT_TRUE(AddTernaryOp(graph));

  testing::FileCheck()
      .check_count("= prim::Ternary", 1, /*exactly*/ true)
      ->check_count("= prim::If", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(AddTernaryOpTest, NoTernaryOpMultipleOutputs) {
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
  EXPECT_FALSE(AddTernaryOp(graph));

  testing::FileCheck()
      .check_count("= prim::Ternary", 0, /*exactly*/ true)
      ->check_count("= prim::If", 1, /*exactly*/ true)
      ->run(*graph);
}

} // namespace jit
} // namespace torch
