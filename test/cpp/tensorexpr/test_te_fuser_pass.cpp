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
    graph(%0 : Float(128:1, device=cpu),
          %1 : Float(128:1, device=cpu)):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Float(128:1, device=cpu) = aten::mul(%0, %1)
      %2 : Float(128:1, device=cpu) = aten::mul(%2.1, %1)
      %3 : Float(128:1, device=cpu) = aten::add_(%2, %1, %12)
      %4 : Float(128:1, device=cpu) = aten::mul(%2, %1)
      %5 : Float(128:1, device=cpu) = aten::add(%2, %4, %12)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // We should not be able to fuse across the in-place operation here.
  testing::FileCheck()
      .check("prim::TensorExprGroup_0")
      ->check("aten::add_")
      ->check("prim::TensorExprGroup_1")
      ->run(*g);
}

void testFuserPass_2() {
  KernelScope kernel_scope;
  const auto graph_string = R"IR(
    graph(%0 : Float(128:1, device=cpu),
          %1 : Float(128:1, device=cpu)):
      %12 : int = prim::Constant[value=1]()
      %a : Float(128:1, device=cpu) = aten::mul(%0, %1)
      %b : Float(128:1, device=cpu) = aten::add(%0, %1, %12)
      %c : Float(128:1, device=cpu) = aten::add_(%b, %1, %12)
      %d : Float(128:1, device=cpu) = aten::mul(%c, %a)
      return (%d))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // We should not be able to fuse across the in-place operation here.
  testing::FileCheck()
      .check("aten::add_")
      ->check("prim::TensorExprGroup_0")
      ->run(*g);
}

void testFuserPass_3() {
  KernelScope kernel_scope;
  const auto graph_string = R"IR(
    graph(%x : Float(128:1, device=cpu),
          %y : Float(128:1, device=cpu)):
      %r : Float(128:1, device=cpu) = aten::mul(%x, %y)
      return (%r))IR";
  {
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // We should not create a fusion group since its size would be too small
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // We should create a fusion group since its size is above the threshold
    testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
  }
}

void testFuserPass_4() {
  KernelScope kernel_scope;
  const auto graph_string = R"IR(
    graph(%a : Float(128:1, device=cpu),
          %b : Float(128:1, device=cpu),
          %c : Float(128:1, device=cpu),
          %d : Float(128:1, device=cpu)):
      %x : Float(128:1, device=cpu) = aten::mul(%a, %b)
      %y : Float(128:1, device=cpu) = aten::mul(%c, %d)
      return (%x, %y))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 1);

  // The %x and %y computations are completely independent and yet we should put
  // them into a single fusion group rather than having two separate ones.
  testing::FileCheck()
      .check("tensorexpr::Group_0")
      ->check_not("tensorexpr::Group_1")
      ->run(*g);
}

} // namespace jit
} // namespace torch
