#include <gtest/gtest.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>

namespace torch {
namespace jit {

namespace {
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}
} // namespace

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

class Quantization : public ::testing::Test {
 public:
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  void SetUp() {
    getTEMustUseLLVMOnCPU() = false;
  }
};

TEST_F(Quantization, Quant) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : int = prim::Constant[value=130]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : Float(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        return (%q.1))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y_expected = at::quantize_per_tensor(x, 0.1f, 130, at::kQUInt8);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  CHECK_EQ(almostEqual(y_expected, y), 1);
#endif
}

TEST_F(Quantization, QuantDequantInt8) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()
        %3 : int = prim::Constant[value=13]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : Float(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(2, 2) = aten::dequantize(%q.1)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 13, at::kQInt8);
  auto y_expected = at::dequantize(q);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  CHECK_EQ(almostEqual(y_expected, y), 1);
#endif
}

TEST_F(Quantization, QuantDequantUInt8) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : int = prim::Constant[value=130]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : Float(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(2, 2) = aten::dequantize(%q.1)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 130, at::kQUInt8);
  auto y_expected = at::dequantize(q);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  CHECK_EQ(almostEqual(y_expected, y), 1);
#endif
}

at::Tensor quantized_add(
    at::Tensor x1,
    at::Tensor x2,
    double scale,
    int64_t zero) {
  const auto qadd_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::add", "")
          .typed<at::Tensor(at::Tensor, at::Tensor, double, int64_t)>();
  return qadd_op.call(x1, x2, scale, zero);
}

TEST_F(Quantization, QuantAddDequantInt8) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()
        %qz1 : int = prim::Constant[value=13]()
        %qs1 : float = prim::Constant[value=0.1]()
        %qz2 : int = prim::Constant[value=13]()
        %qs2 : float = prim::Constant[value=0.1]()
        %qza : int = prim::Constant[value=13]()
        %qsa : float = prim::Constant[value=0.1]()
        %q1 : Float(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : Float(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        %qa : Float(2, 2) = quantized::add(%q1, %q2, %qsa, %qza)
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQInt8);
  auto qa = quantized_add(q1, q2, 0.1f, 13);
  auto y_expected = at::dequantize(qa);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1, x2};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  if (!almostEqual(y_expected, y)) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  // CHECK_EQ(almostEqual(y_expected, y), 1);
#endif
}

TEST_F(Quantization, QuantAddDequantUInt8) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %qz1 : int = prim::Constant[value=130]()
        %qs1 : float = prim::Constant[value=0.1]()
        %qz2 : int = prim::Constant[value=140]()
        %qs2 : float = prim::Constant[value=0.2]()
        %qza : int = prim::Constant[value=150]()
        %qsa : float = prim::Constant[value=0.3]()
        %q1 : Float(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : Float(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        %qa : Float(2, 2) = quantized::add(%q1, %q2, %qsa, %qza)
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 130, at::kQUInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.2f, 140, at::kQUInt8);
  auto qa = quantized_add(q1, q2, 0.3f, 150);
  auto y_expected = at::dequantize(qa);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1, x2};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  if (!almostEqual(y_expected, y)) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  // CHECK_EQ(almostEqual(y_expected, y), 1);
#endif
}

} // namespace jit
} // namespace torch
