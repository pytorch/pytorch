#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <sstream>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void testFuserPass_1() {
  KernelScope kernel_scope;
  const auto graph_string = R"IR(
    graph(%0 : Float(128:1),
          %1 : Float(128:1)):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Float(128:1) = aten::mul(%0, %1)
      %2 : Float(128:1) = aten::mul(%2.1, %1)
      %3 : Float(128:1) = aten::add_(%2, %1, %12)
      %4 : Float(128:1) = aten::mul(%2, %1)
      %5 : Float(128:1) = aten::add(%2, %4, %12)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // We should not be able to fuse across the in-place operation here.
  testing::FileCheck()
      .check("tensorexpr::Group_0")
      ->check("aten::add_")
      ->check("tensorexpr::Group_1")
      ->run(*g);
}

void testFuserPass_2() {
  KernelScope kernel_scope;
  const auto graph_string = R"IR(
    graph(%0 : Float(128:1),
          %1 : Float(128:1)):
      %12 : int = prim::Constant[value=1]()
      %a : Float(128:1) = aten::mul(%0, %1)
      %b : Float(128:1) = aten::add(%0, %1, %12)
      %c : Float(128:1) = aten::add_(%b, %1, %12)
      %d : Float(128:1) = aten::mul(%c, %a)
      return (%d))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);
  std::cerr << *g << "\n";

  // We should not be able to fuse across the in-place operation here.
  testing::FileCheck()
      .check("aten::add_")
      ->check("tensorexpr::Group_0")
      ->run(*g);
}

} // namespace jit
} // namespace torch
