#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <memory>

namespace torch {
namespace jit {

TEST(OpReplacementTest, ReplaceDivInSimpleFunction) {
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)
            %3 : Tensor  = aten::div(%2, %1)
            return (%3))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->set_op_version(2);
  ApplyOldOpsUpgraders(g);
  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::div(%2, %1)", 1, /*exactly=*/true)
      ->check_count("aten::div(%2, %1, %4)", 1, /*exactly=*/true)
      ->run(*g);

  EXPECT_TRUE(g->get_op_version().has_value());
  EXPECT_EQ(g->get_op_version().value(), 4);
}

TEST(OpReplacementTest, ReplaceDivInNestedFunction) {
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor,
              %8 : bool):
            %9 : bool = prim::Constant[value=1]()
            %7 : bool = prim::If(%8)
                block0():
                    -> (%9)
                block1():
                    %2 : Tensor = aten::add(%0, %1)
                    %3 : Tensor  = aten::div(%2, %1)
                    %4 : Tensor = aten::add(%3, %0)
                    %10 : bool = aten::is_floating_point(%4)
                    -> (%10)
            return (%7))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->set_op_version(2);
  ApplyOldOpsUpgraders(g);
  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::add", 2, false)
      ->run(*g);

  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::div", 2, false)
      ->run(*g);

  EXPECT_TRUE(g->get_op_version().has_value());
  EXPECT_EQ(g->get_op_version().value(), 4);
}

TEST(OpReplacementTest, ReplaceTestSubcmulInSimpleFunction) {
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %3 : int = prim::Constant[value=1]()
            %2 : Tensor = aten::_test_serialization_subcmul(%0, %1, %3)
            return (%2))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->set_op_version(2);
  ApplyOldOpsUpgraders(g);
  testing::FileCheck()
      .check_count("aten::mul", 1, false)
      ->run(*g);

  testing::FileCheck()
      .check_count("aten::sub", 1, false)
      ->run(*g);

  EXPECT_TRUE(g->get_op_version().has_value());
  EXPECT_EQ(g->get_op_version().value(), 3);
}

} // namespace jit
} // namespace torch
