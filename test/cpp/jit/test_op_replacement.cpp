#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
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
                            return (%35))IR"}});

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
  ReplaceOldOperatorsWithUpgraders(g);
  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::div(%2, %1)", 1, /*exactly=*/true)
      ->check_count("aten::div(%2, %1, %4)", 1, /*exactly=*/true)
      ->run(*g);
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
  UpgraderEntry test_entry{
      3,
      "_test_serialization_subcmul_0_2",
      "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"};
  test_only_add_entry("aten::_test_serialization_subcmul", test_entry);
  torch::jit::parseIR(graph_string, g.get());
  g->set_op_version(2);
  ReplaceOldOperatorsWithUpgraders(g);
  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::div", 2, /*exactly=*/true)
      ->run(*g);
  test_only_remove_entry("aten::_test_serialization_subcmul");
  test_only_remove_upgraders(test_upgraders);
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
  ReplaceOldOperatorsWithUpgraders(g);
  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::add", 2, false)
      ->run(*g);

  testing::FileCheck()
      .check("prim::If")
      ->check_count("aten::div", 2, false)
      ->run(*g);
  test_only_remove_upgraders(test_upgraders);
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
  UpgraderEntry test_entry{
      3,
      "_test_serialization_subcmul_0_2",
      "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"};
  test_only_add_entry("aten::_test_serialization_subcmul", test_entry);
  torch::jit::parseIR(graph_string, g.get());
  g->set_op_version(2);
  ReplaceOldOperatorsWithUpgraders(g);
  testing::FileCheck().check_count("aten::mul", 1, false)->run(*g);

  testing::FileCheck().check_count("aten::sub", 1, false)->run(*g);

  test_only_remove_upgraders(test_upgraders);
  test_only_remove_entry("aten::_test_serialization_subcmul");
}

} // namespace jit
} // namespace torch
