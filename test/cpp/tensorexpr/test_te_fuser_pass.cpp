#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/mem_arena.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <sstream>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

struct WithCPUFuser {
  WithCPUFuser(bool val = true)
      : cpuFuserEnabled(canFuseOnCPU()), parallel(texprParallelCPUEnabled()) {
    overrideCanFuseOnCPU(val);
    setTexprParallelCPUEnabled(true);
  }

  ~WithCPUFuser() {
    overrideCanFuseOnCPU(cpuFuserEnabled);
    setTexprParallelCPUEnabled(parallel);
  }

  bool cpuFuserEnabled;
  bool parallel;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_1) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%0 : Float(128, strides=[1], device=cpu),
          %1 : Float(128, strides=[1], device=cpu)):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Float(128, strides=[1], device=cpu) = aten::mul(%0, %1)
      %2 : Float(128, strides=[1], device=cpu) = aten::mul(%2.1, %1)
      %3 : Float(128, strides=[1], device=cpu) = aten::add_(%2, %1, %12)
      %4 : Float(128, strides=[1], device=cpu) = aten::mul(%2, %1)
      %5 : Float(128, strides=[1], device=cpu) = aten::add(%2, %4, %12)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // We should not be able to fuse across the in-place operation here.
  testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("aten::add_")
      ->check("prim::TensorExprGroup_")
      ->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_2) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%0 : Float(128, strides=[1], device=cpu),
          %1 : Float(128, strides=[1], device=cpu)):
      %12 : int = prim::Constant[value=1]()
      %a : Float(128, strides=[1], device=cpu) = aten::mul(%0, %1)
      %b : Float(128, strides=[1], device=cpu) = aten::add(%0, %1, %12)
      %c : Float(128, strides=[1], device=cpu) = aten::add_(%b, %1, %12)
      %d : Float(128, strides=[1], device=cpu) = aten::mul(%c, %a)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_3) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Float(128, strides=[1], device=cpu),
          %y : Float(128, strides=[1], device=cpu)):
      %r : Float(128, strides=[1], device=cpu) = aten::mul(%x, %y)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_0DimInput) {
  const auto graph_string = R"IR(
    graph(%x : Float(device=cuda),
          %y : Float(device=cuda)):
      %one : int = prim::Constant[value=1]()
      %a : Float(device=cuda) = aten::mul(%x, %y)
      %b : Float(device=cuda) = aten::add(%x, %a, %one)
      return (%b))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // We should not fuse 0-dim tensors
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_UnfusibleDevice) {
  WithCPUFuser cf(false);
  const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(10, strides=[1], device=cpu)):
      %a : Float(10, strides=[1], device=cpu) = aten::mul(%x, %y)
      return (%a))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 1);

  // Test that we're not starting fusion groups from nodes with unfusible device
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_UnknownShapes) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Tensor,
          %y : Tensor):
      %a : Tensor = aten::mul(%x, %y)
      %b : Tensor = aten::mul(%x, %a)
      return (%b))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g);

  // Test that we're not generating fusion groups when shapes are not known
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_Multidevice) {
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      return (%cat))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // We should be able to fuse this
    testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
  }
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cuda:0),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      return (%cat))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // We should not fuse this aten::cat since its inputs are from different
    // devices
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(10, strides=[1], device=cuda:0)):
      %dim : int = prim::Constant[value=0]()
      %xy_list : Tensor[] = prim::ListConstruct(%x, %y)
      %xy_cat : Float(30, strides=[1], device=cpu) = aten::cat(%xy_list, %dim)
      %r : Float(30, strides=[1], device=cpu) = aten::mul(%xy_cat, %z)
      return (%r))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // Test that we check device before merging one node (cat) into another
    // (mul)
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(10, strides=[1], device=cuda:0)):
      %z2 : Tensor = aten::mul(%z, %z)
      %dim : int = prim::Constant[value=0]()
      %xy_list : Tensor[] = prim::ListConstruct(%x, %y, %z2)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xy_list, %dim)
      return (%cat))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // Test that we check device before merging one node (mul) into another
    // (cat)
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cuda:0)):
      %r : Float(10, strides=[1], device=cpu) = aten::mul(%x, %y)
      return (%r))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // We should not fuse this graph since its inputs are from different devices
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    WithCPUFuser cf;
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cuda:0),
          %y : Float(20, strides=[1], device=cuda:1),
          %z : Float(20, strides=[1], device=cpu)):
      %x2 : Float(10, strides=[1], device=cpu) = aten::mul(%x, %x)
      %y2 : Float(10, strides=[1], device=cpu) = aten::mul(%y, %y)
      %z2 : Float(10, strides=[1], device=cpu) = aten::mul(%z, %z)
      return (%x2, %y2, %z2))IR";
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    g->lint();
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // We should not fuse these two computations since they use different
    // devices
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_MergeGroups) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%a : Float(128, strides=[1], device=cpu),
          %b : Float(128, strides=[1], device=cpu)):
      %x : Float(128, strides=[1], device=cpu) = aten::mul(%a, %a)
      %y : Float(128, strides=[1], device=cpu) = aten::mul(%b, %b)
      return (%x, %y))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 1);

  // The %x and %y computations are completely independent and yet we should put
  // them into a single fusion group rather than having two separate ones.
  testing::FileCheck()
      .check("= prim::TensorExprGroup_")
      ->check_not("= prim::TensorExprGroup_")
      ->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_UnknownShapesIgnored) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Float(device=cpu),
          %y : Float(device=cpu)):
      %a : Float(device=cpu) = aten::mul(%x, %y)
      %b : Float(device=cpu) = aten::mul(%x, %a)
      return (%b))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 2, /* disable_shape_checks= */ true);

  // Test that we are generating fusion groups even though shapes are not known
  testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_IgnoreUnknownShapeAtStart) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Bool(8, strides=[1], device=cpu),
          %y : Bool(8, strides=[1], device=cpu)):
      %a : Bool(8, strides=[1], device=cpu) = aten::__and__(%x, %y)
      %b : Tensor = aten::__or__(%a, %y)
      return (%b)
    )IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 2);
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_Where) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Float(8, strides=[1], device=cpu),
          %y : Float(8, strides=[1], device=cpu),
          %z : Float(8, strides=[1], device=cpu)):
      %cond : Bool(8, strides=[1], device=cpu) = aten::eq(%x, %y)
      %b : Float(8, strides=[1], device=cpu) = aten::where(%cond, %y, %z)
      return (%b)
    )IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 2);
  testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TEFuserPass, FuserPass_WhereList) {
  WithCPUFuser cf;
  const auto graph_string = R"IR(
    graph(%x : Float(8, strides=[1], device=cpu),
          %y : Float(8, strides=[1], device=cpu),
          %z : Float(8, strides=[1], device=cpu)):
      %cond : Bool(8, strides=[1], device=cpu) = aten::eq(%x, %y)
      %b : Tensor[] = aten::where(%cond)
      return (%b)
    )IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  FuseTensorExprs(g, /* min_group_size= */ 2);
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

} // namespace jit
} // namespace torch
