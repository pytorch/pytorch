#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include <limits>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

class GraphOpt : public ::testing::Test {
 public:
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  void SetUp() {
    old_cat_wo_conditionals_ = getCatWoConditionals();
    getCatWoConditionals() = true;
  }

  void TearDown() {
    getCatWoConditionals() = old_cat_wo_conditionals_;
  }

 private:
  bool old_cat_wo_conditionals_;
};

TEST_F(GraphOpt, OptimizeCat) {
#ifdef TORCH_ENABLE_LLVM
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

  TensorExprKernel kernel(g);

  // The `aten::log` op must be moved to the inputs of `aten::cat`.
  testing::FileCheck()
      .check("aten::log")
      ->check("aten::log")
      ->check("aten::log")
      ->check("aten::cat")
      ->check_not("aten::log")
      ->run(*kernel.graph());

  auto x = at::rand({10}, at::kFloat);
  auto y = at::rand({20}, at::kFloat);
  auto z = at::rand({30}, at::kFloat);
  auto ref = at::log(at::cat({x, y, z}, 0));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));
#endif
}

TEST_F(GraphOpt, OptimizeCat2) {
#ifdef TORCH_ENABLE_LLVM
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

  auto x = at::rand({10}, at::kFloat);
  auto y = at::rand({20}, at::kFloat);
  auto z = at::rand({30}, at::kFloat);
  auto ref = at::tanh(at::log(at::cat({x, y, z}, 0)));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));
#endif
}

TEST_F(GraphOpt, OptimizeCat3) {
#ifdef TORCH_ENABLE_LLVM
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

  auto a = at::rand({60}, at::kFloat);
  auto x = at::rand({10}, at::kFloat);
  auto y = at::rand({20}, at::kFloat);
  auto z = at::rand({30}, at::kFloat);
  auto ref = at::tanh(at::cat({x, y, z}, 0)) * a;

  std::vector<at::Tensor> inputs = {a, x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));
#endif
}

TEST_F(GraphOpt, OptimizeCatWithTypePromotionInUser) {
#ifdef TORCH_ENABLE_LLVM
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

  auto x = at::randint(std::numeric_limits<int>::max(), {10}, at::kInt);
  auto y = at::randint(std::numeric_limits<int>::max(), {20}, at::kInt);
  auto z = at::randint(std::numeric_limits<int>::max(), {30}, at::kInt);
  auto ref = at::tanh(at::cat({x, y, z}, 0));

  std::vector<at::Tensor> inputs = {x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  kernel.run(stack);
  auto out = stack[0].toTensor();
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.dtype(), ref.dtype());
  ASSERT_TRUE(at::allclose(out, ref));
#endif
}

TEST_F(GraphOpt, OptimizeCatWithTypePromotionInCat) {
#ifdef TORCH_ENABLE_LLVM
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

  TensorExprKernel kernel(g);

  // No transformation should have happened because the `aten::cat` op performs
  // type promotion. This case is currently not handled.
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::log")
      ->check_not("aten::cat")
      ->check_not("aten::log")
      ->run(*kernel.graph());
#endif
}

TEST_F(GraphOpt, OptimizeCatNoSingleTensorElementwiseOp) {
#ifdef TORCH_ENABLE_LLVM
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

  TensorExprKernel kernel(g);

  // No transformation is expected since the consumers of cat are not
  // single-tensor element-wise ops.
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::mul")
      ->check_not("aten::cat")
      ->check_not("aten::mul")
      ->run(*kernel.graph());
#endif
}

TEST_F(GraphOpt, OptimizeCatNoSingleTensorElementwiseOp2) {
#ifdef TORCH_ENABLE_LLVM
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
#endif
}

TEST_F(GraphOpt, AOTGraphPrepPasses) {
  const auto graph_string = R"IR(
    graph(%x, %y, %z, %i : int):
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      return (%xyz_list, %i))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  removeGraphOutput(g, 1);
  replaceListOutputWithTuple(g);
  LowerAllTuples(g);

  testing::FileCheck().check("return (%x, %y, %z)")->run(*g);
}

TEST_F(GraphOpt, DecomposeQuantizationAddandMul) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
    graph(%x : Float(8, 16, strides=[16, 1], device=cpu)):
      %1 : int = prim::Constant[value=0]()
      %2 : float = prim::Constant[value=0.86077326536178589]()
      %3 : int = prim::Constant[value=63]()
      %4 : float = prim::Constant[value=0.16712260246276855]()
      %5 : int = prim::Constant[value=60]()
      %6 : float = prim::Constant[value=0.08735954761505127]()
      %7 : int = prim::Constant[value=13]()
      %8 : Long(device=cpu) = prim::Constant[value={59}]()
      %9 : Float(device=cpu) = prim::Constant[value={0.0421975}]()
      %10 : QUInt8(8, 16, strides=[16, 1], device=cpu) = aten::quantize_per_tensor(%x, %9, %8, %7)
      %11 : QUInt8(8, 16, strides=[16, 1], device=cpu) = quantized::add(%10, %10, %6, %5)
      %12 : QUInt8(8, 16, strides=[16, 1], device=cpu) = quantized::add(%11, %11, %4, %3)
      %mul_1.1 : QUInt8(8, 16, strides=[16, 1], device=cpu) = quantized::mul(%12, %12, %2, %1)
      %14 : Float(8, 16, strides=[16, 1], device=cpu) = aten::dequantize(%mul_1.1)
      return (%14))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get(), true);
  g->lint();

  TensorExprKernel kernel(g);

  // Quantized::add should be decomposed to be
  //   aten::dequantize/aten::add/aten::quantize_per_tensor
  // Similar for quantized::mul
  testing::FileCheck()
      .check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->check("aten::add")
      ->check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->check("aten::add")
      ->check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->check("aten::mul")
      ->check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->run(*kernel.graph());
#endif
}

TEST_F(GraphOpt, DecomposeQuantizationAddRelu) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
    graph(%x : Float(8, 16, strides=[16, 1], device=cpu)):
      %1 : float = prim::Constant[value=0.20982688665390015]()
      %2 : int = prim::Constant[value=0]()
      %3 : float = prim::Constant[value=0.045946117490530014]()
      %4 : int = prim::Constant[value=13]()
      %5 : Long(device=cpu) = prim::Constant[value={59}]()
      %6 : Float(device=cpu) = prim::Constant[value={0.0421975}]()
      %7 : QUInt8(8, 16, strides=[16, 1], device=cpu) = aten::quantize_per_tensor(%x, %6, %5, %4)
      %8 : QUInt8(8, 16, strides=[16, 1], device=cpu) = quantized::add_relu(%7, %7, %3, %2)
      %mul_1.1 : QUInt8(8, 16, strides=[16, 1], device=cpu) = quantized::mul(%8, %8, %1, %2)
      %10 : Float(8, 16, strides=[16, 1], device=cpu) = aten::dequantize(%mul_1.1)
      return (%10))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get(), true);
  g->lint();

  TensorExprKernel kernel(g);

  // Quantized::add_relu should be decomposed to be
  //   aten::dequantize/aten::add/aten::relu/aten::quantize_per_tensor
  // Quantized::mul should be decomposed to be
  //   aten::dequantize/aten::mul/aten::quantize_per_tensor
  testing::FileCheck()
      .check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->check("aten::add")
      ->check("aten::relu")
      ->check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->check("aten::mul")
      ->check("aten::quantize_per_tensor")
      ->check("aten::dequantize")
      ->run(*kernel.graph());
#endif
}

} // namespace jit
} // namespace torch
