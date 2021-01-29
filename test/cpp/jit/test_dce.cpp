#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
TEST(EliminateDeadCodeTest, Basic) {
  auto graph = std::make_shared<Graph>();

  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  // We want to check that b[0] and b are properly marked as live and thus not
  // DCE'd.
  const std::string input =
      R"IR(
graph():
  %48 : None = prim::Constant()
  %50 : bool = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=2]()
  %12 : int = prim::Constant[value=1]()
  %24 : int = prim::Constant[value=3]()
  %31 : int = prim::Constant[value=0]()
  %2 : int[] = prim::ListConstruct(%0, %0)
  %a.1 : Tensor = prim::MakeTestTensor()
  %14 : int[] = prim::ListConstruct(%12)
  %tot.1 : Tensor = prim::MakeTestTensor()
  %tot : Tensor = prim::Loop(%24, %50, %tot.1)
    block0(%i : int, %tot.6 : Tensor):
      %33 : Tensor = aten::select(%a.1, %31, %31)
      %35 : Tensor = aten::select(%33, %31, %31)
      # CHECK: add_
      %tot.3 : Tensor = aten::add_(%tot.6, %35, %12)
      %b.1 : Tensor = aten::select(%a.1, %31, %31)
      %44 : Tensor = aten::select(%b.1, %31, %31)
      # CHECK: add_
      %46 : Tensor = aten::add_(%44, %12, %12)
      -> (%50, %tot.3)
  return (%tot)
)IR";
  parseIR(input, graph.get());
  EliminateDeadCode(graph);
  // Check that dead code elimin
  testing::FileCheck().run(input, *graph);
}
} // namespace jit
} // namespace torch
