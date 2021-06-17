#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include <limits>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCat) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::log(%cat)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // The `aten::log` op must be moved to the inputs of `aten::cat`.
  testing::FileCheck()
      .check("aten::log")
      ->check("aten::log")
      ->check("aten::log")
      ->check("aten::cat")
      ->check_not("aten::log")
      ->run(*kernel.graph());

  auto x = at::rand({10}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y = at::rand({20}, TensorOptions(kCPU).dtype(at::kFloat));
  auto z = at::rand({30}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::log(at::cat({x, y, z}, 0));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));

  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCat2) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::log(%cat)
      %6 : Float(60, strides=[1], device=cpu) = aten::tanh(%5)
      return (%6))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // The `aten::log` and `aten::tanh` ops must be moved to the inputs of
  // `aten::cat`.
  testing::FileCheck()
      .check("aten::log")
      ->check("aten::log")
      ->check("aten::log")
      ->check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::cat")
      ->check_not("aten::log")
      ->check_not("aten::tanh")
      ->run(*kernel.graph());

  auto x = at::rand({10}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y = at::rand({20}, TensorOptions(kCPU).dtype(at::kFloat));
  auto z = at::rand({30}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::tanh(at::log(at::cat({x, y, z}, 0)));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));

  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCat3) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%a : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::tanh(%cat)
      %6 : Float(60, strides=[1], device=cpu) = aten::mul(%a, %5)
      return (%6))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // The `aten::tanh` op must be moved to the inputs of `aten::cat`.
  // But the `aten::mul` op must not be moved since it is not a single-tensor
  // op (it has 2 tensor inputs).
  testing::FileCheck()
      .check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::cat")
      ->check("aten::mul")
      ->check_not("aten::tanh")
      ->run(*kernel.graph());

  auto a = at::rand({60}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x = at::rand({10}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y = at::rand({20}, TensorOptions(kCPU).dtype(at::kFloat));
  auto z = at::rand({30}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::tanh(at::cat({x, y, z}, 0)) * a;

  std::vector<at::Tensor> inputs = {a, x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));

  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCatWithTypePromotionInUser) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%x : Int(10, strides=[1], device=cpu),
          %y : Int(20, strides=[1], device=cpu),
          %z : Int(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Int(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::tanh(%cat)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // The `aten::tanh` op must be moved to the inputs of `aten::cat`.
  // The scalar type of the inputs to `cat` should now be `Float` since they
  // are the result of `tanh` which does the type promotion.
  testing::FileCheck()
      .check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::cat")
      ->check_not("aten::tanh")
      ->run(*kernel.graph());

  auto x = at::randint(
      std::numeric_limits<int>::max(),
      {10},
      TensorOptions(kCPU).dtype(at::kInt));
  auto y = at::randint(
      std::numeric_limits<int>::max(),
      {20},
      TensorOptions(kCPU).dtype(at::kInt));
  auto z = at::randint(
      std::numeric_limits<int>::max(),
      {30},
      TensorOptions(kCPU).dtype(at::kInt));
  auto ref = at::tanh(at::cat({x, y, z}, 0));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));

  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCatWithTypePromotionInCat) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Double(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Double(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Double(60, strides=[1], device=cpu) = aten::log(%cat)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // No transformation should have happened because the `aten::cat` op performs
  // type promotion. This case is currently not handled.
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::log")
      ->check_not("aten::cat")
      ->check_not("aten::log")
      ->run(*kernel.graph());

  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCatNoSingleTensorElementwiseOp) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%0 : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::mul(%0, %cat)
      return (%5))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // No transformation is expected since the consumers of cat are not
  // single-tensor element-wise ops.
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::mul")
      ->check_not("aten::cat")
      ->check_not("aten::mul")
      ->run(*kernel.graph());
  getCatWoConditionals() = old_flag;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GraphOpt, OptimizeCatNoSingleTensorElementwiseOp2) {
  bool old_flag = getCatWoConditionals();
  getCatWoConditionals() = true;
  const auto graph_string = R"IR(
    graph(%0 : Float(60, strides=[1], device=cpu),
          %1 : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %one : int = prim::Constant[value=1]()
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::mul(%0, %cat)
      %6 : Float(60, strides=[1], device=cpu) = aten::add(%5, %1, %one)
      return (%6))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  KernelScope kernel_scope;
  TensorExprKernel kernel(g);

  // No transformation is expected since the consumers of cat are not
  // single-tensor element-wise ops.
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::mul")
      ->check("aten::add")
      ->check_not("aten::cat")
      ->check_not("aten::mul")
      ->check_not("aten::add")
      ->run(*kernel.graph());
  getCatWoConditionals() = old_flag;
}

} // namespace jit
} // namespace torch
