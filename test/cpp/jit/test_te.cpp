#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <iostream>

namespace torch {
namespace jit {

TEST(TETest, RemoveProfiling) {
  auto g = std::make_shared<Graph>();
  const auto graph_string = R"IR(
    graph(%a : Tensor,
          %b : bool):
      %1 : None = prim::Constant()
      %2 : Tensor? = prim::If(%b)
        block0():
          %3 : Tensor? = prim::profile[profiled_type=Tensor, seen_none=0](%1)
          -> (%3)
        block1():
          %4 : Tensor = prim::profile[profiled_type=Tensor, seen_none=0](%a)
          -> (%4)
      return (%2))IR";
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  RemoveProfileNodesAndSpecializeTypes(g);
  g->lint();

  testing::FileCheck()
      .check("prim::Constant")
      ->check("prim::If")
      ->check("block")
      ->check("block")
      ->check("return")
      ->run(*g);
}
} // namespace jit
} // namespace torch
