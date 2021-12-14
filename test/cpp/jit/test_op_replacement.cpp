#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>
#include <memory>

namespace torch {
namespace jit {

std::unordered_map<std::string, std::string> test_upgraders(
    {{"_test_serialization_subcmul_0_2", R"IR(graph(%self.1 : Tensor,
                                                    %other.1 : Tensor,
                                                    %alpha.1 : Union(float, int)):
                                                %7 : int = prim::Constant[value=1]()
                                                %6 : Tensor = aten::mul(%self.1, %alpha.1) # torch/jit/operator_upgraders.py:18:20
                                                %8 : Tensor = aten::sub(%other.1, %6, %7) # torch/jit/operator_upgraders.py:18:11
                                                return (%8))IR"},
     {"div_Tensor_0_3", R"IR(graph(%self.1 : Tensor,
                                  %other.1 : Tensor):
                            %32 : str = prim::Constant[value="trunc"]()
                            %6 : bool = prim::Constant[value=1]()
                            %4 : bool = aten::is_floating_point(%self.1)
                            %11 : bool = prim::If(%4)
                                block0():
                                    -> (%6)
                                block1():
                                    %9 : bool = aten::is_floating_point(%other.1)
                                    -> (%9)
                            %35 : Tensor = prim::If(%11)
                                block0():
                                    %36 : Tensor = aten::div(%self.1, %other.1)
                                    -> (%36)
                                block1():
                                    %37 : Tensor = aten::div(%self.1, %other.1, %32)
                                    -> (%37)
                            return (%35))IR"},
     {"div_Scalar_0_3", R"IR(graph(%self.1 : Tensor,
                                %other.1 : Scalar):
                            %41 : str = prim::Constant[value=\"trunc\"]()
                            %6 : bool = prim::Constant[value=1]()
                            %4 : bool = aten::is_floating_point(%self.1)
                            %9 : bool = prim::If(%4)
                                block0():
                                    -> (%6)
                                block1():
                                    %8 : bool = prim::isinstance[types=[float]](%other.1)
                                    -> (%8)
                            %44 : Tensor = prim::If(%9) # torch/jit/operator_upgraders.py:21:4
                                block0():
                                    %45 : Tensor = aten::div(%self.1, %other.1) # torch/jit/operator_upgraders.py:22:15
                                    -> (%45)
                                block1():
                                    %other.9 : Union[complex, int] = prim::unchecked_cast(%other.1)
                                    %46 : Tensor = aten::div(%self.1, %other.9, %41) # torch/jit/operator_upgraders.py:23:11
                                    -> (%46)
                            return (%44))IR"},
     {"div_out_0_3", R"IR(graph(%self.1 : Tensor,
                            %other.1 : Tensor,
                            %out.1 : Tensor):
                        %41 : str = prim::Constant[value="trunc"]() # torch/jit/operator_upgraders.py:33:44
                        %7 : bool = prim::Constant[value=1]() # torch/jit/operator_upgraders.py:31:8
                        %5 : bool = aten::is_floating_point(%self.1) # torch/jit/operator_upgraders.py:31:8
                        %12 : bool = prim::If(%5) # torch/jit/operator_upgraders.py:31:8
                            block0():
                                -> (%7)
                            block1():
                            %10 : bool = aten::is_floating_point(%other.1) # torch/jit/operator_upgraders.py:31:36
                                -> (%10)
                        %18 : bool = prim::If(%12) # torch/jit/operator_upgraders.py:31:8
                            block0():
                                -> (%7)
                            block1():
                                %16 : bool = aten::is_floating_point(%out.1) # torch/jit/operator_upgraders.py:31:65
                                -> (%16)
                        %44 : Tensor = prim::If(%18) # torch/jit/operator_upgraders.py:31:4
                            block0():
                                %45 : Tensor = aten::div(%self.1, %other.1, %out.1) # torch/jit/operator_upgraders.py:32:15
                                -> (%45)
                            block1():
                                %46 : Tensor = aten::div(%self.1, %other.1, %41, %out.1) # torch/jit/operator_upgraders.py:33:11
                                -> (%46)
                        return (%44))IR"}});

TEST(OpReplacementTest, ReplaceDivInSimpleFunction) {
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)
            %3 : Tensor  = aten::div(%2, %1)
            return (%3))IR";
  auto g = std::make_shared<Graph>();
  test_only_populate_upgraders(test_upgraders);
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

TEST(OpReplacementTest, ReplaceTwoOpsInSimpleFunction) {
  const auto graph_string = R"IR(
        graph(%0 : Tensor,
              %1 : Tensor):
            %2 : Tensor = aten::add(%0, %1)
            %3 : Tensor  = aten::div(%2, %1)
            %4 : int = prim::Constant[value=1]()
            %5: Tensor = aten::_test_serialization_subcmul(%0, %1, %4)
            return (%3, %5))IR";
  auto g = std::make_shared<Graph>();
  test_only_populate_upgraders(test_upgraders);
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
  test_only_populate_upgraders(test_upgraders);
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
  test_only_populate_upgraders(test_upgraders);
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
