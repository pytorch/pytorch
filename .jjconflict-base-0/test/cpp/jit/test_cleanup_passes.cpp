#include <gtest/gtest.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

TEST(CleanupPassTest, Basic) {
  // Tests stability of clean up passes when dealing with constant pooling
  // and constant propagation.
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%cond.1 : Tensor,
      %suffix.1 : str):
  %3 : bool = aten::Bool(%cond.1) # o.py:6:7
  %25 : str = prim::If(%3) # o.py:6:4
    block0():
      %a.1 : str = prim::Constant[value="same string"]()
      %b.1 : str = prim::Constant[value=" with a twist"]()
      %7 : str = aten::add(%a.1, %b.1)
      %11 : str = aten::add(%7, %suffix.1) # o.py:10:15
      -> (%11)
    block1():
      %c.1 : str = prim::Constant[value="same string"]()
      %d.1 : str = prim::Constant[value=" with a twist"]()
      %12 : str = aten::add(%c.1, %d.1)
      -> (%12)
  return (%25)
  )IR",
      &*graph);
  runCleanupPasses(graph);
  testing::FileCheck()
      .check_count(
          "prim::Constant[value=\"same string with a twist\"]",
          1,
          /*exactly=*/true)
      ->run(*graph);

  auto graph_after_pass_once = graph->toString();
  runCleanupPasses(graph);
  auto graph_after_pass_twice = graph->toString();
  ASSERT_EQ(graph_after_pass_once, graph_after_pass_twice);
}
} // namespace jit
} // namespace torch
