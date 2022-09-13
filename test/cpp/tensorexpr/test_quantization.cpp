#include <gtest/gtest.h>

#include <ATen/native/quantized/PackedParams.h>
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
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;
using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

class Quantization : public ::testing::Test {
 public:
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  void SetUp() {
    getTEMustUseLLVMOnCPU() = false;
  }
};

TEST_F(Quantization, QuantDequantInt8) {
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()
        %3 : int = prim::Constant[value=13]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : QInt8(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
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

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : int = prim::Constant[value=122]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(2, 2) = aten::dequantize(%q.1)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = 2 * at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 122, at::kQUInt8);
  auto y_expected = at::dequantize(q);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantDequantUInt8_NLC) {
  const auto graph_string = R"IR(
      graph(%x.1 : Float(1, 2, 2, strides=[4, 1, 2], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : int = prim::Constant[value=122]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : QUInt8(1, 2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(1, 2, 2) = aten::dequantize(%q.1)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = 2 * at::rand({1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  x.unsafeGetTensorImpl()->set_sizes_and_strides(
      std::initializer_list<int64_t>{1, 2, 2}, {4, 1, 2});
  auto q = at::quantize_per_tensor(x, 0.1f, 122, at::kQUInt8);
  auto y_expected = at::dequantize(q);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
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
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()
        %qz1 : int = prim::Constant[value=13]()
        %qs1 : float = prim::Constant[value=0.1]()
        %qz2 : int = prim::Constant[value=13]()
        %qs2 : float = prim::Constant[value=0.1]()
        %qza : int = prim::Constant[value=13]()
        %qsa : float = prim::Constant[value=0.1]()
        %q1 : QInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : QInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        %qa : QInt8(2, 2) = quantized::add(%q1, %q2, %qsa, %qza)
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
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantAddDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %qz1 : int = prim::Constant[value=13]()
        %qs1 : float = prim::Constant[value=0.1]()
        %qz2 : int = prim::Constant[value=13]()
        %qs2 : float = prim::Constant[value=0.1]()
        %qza : int = prim::Constant[value=13]()
        %qsa : float = prim::Constant[value=0.1]()
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : QUInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        %qa : QUInt8(2, 2) = quantized::add(%q1, %q2, %qsa, %qza)
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQUInt8);
  auto qa = quantized_add(q1, q2, 0.1f, 13);
  auto y_expected = at::dequantize(qa);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1, x2};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantSigmoidDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %qz1 : int = prim::Constant[value=13]()
        %qs1 : float = prim::Constant[value=0.1]()
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %qa : QUInt8(2, 2) = aten::sigmoid(%q1)
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);
  auto qs = at::sigmoid(q1);
  auto y_expected = at::dequantize(qs);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "qs:\n" << qs << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

at::Tensor quantized_mul(
    at::Tensor x1,
    at::Tensor x2,
    double scale,
    int64_t zero) {
  const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::mul", "")
          .typed<at::Tensor(at::Tensor, at::Tensor, double, int64_t)>();
  return op.call(x1, x2, scale, zero);
}

TEST_F(Quantization, QuantMulDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %qz1 : int = prim::Constant[value=13]()
        %qs1 : float = prim::Constant[value=0.1]()
        %qz2 : int = prim::Constant[value=13]()
        %qs2 : float = prim::Constant[value=0.1]()
        %qza : int = prim::Constant[value=13]()
        %qsa : float = prim::Constant[value=0.1]()
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : QUInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        %qa : QUInt8(2, 2) = quantized::mul(%q1, %q2, %qsa, %qza)
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQUInt8);
  auto qa = quantized_mul(q1, q2, 0.1f, 13);
  auto y_expected = at::dequantize(qa);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1, x2};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantUpsampleNearst2dDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 4, 4, strides=[16, 16, 4, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %4 : NoneType = prim::Constant()
        %3 : int[] = prim::Constant[value=[6, 6]]()
        %qz : int = prim::Constant[value=13]()
        %qs : float = prim::Constant[value=0.1]()
        %q : QUInt8(1, 1, 4, 4) = aten::quantize_per_tensor(%x, %qs, %qz, %2)
        %qu : QUInt8(1, 1, 6, 6) = aten::upsample_nearest2d(%q, %3, %4)
        %6 : Float(1, 1, 6, 6) = aten::dequantize(%qu)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 1, 4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 13, at::kQUInt8);
  auto qu = at::upsample_nearest2d(q, {6, 6});
  auto y_expected = at::dequantize(qu);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "q:\n" << q << std::endl;
    std::cout << "qu:\n" << qu << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, UpsampleNearst2d) {
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu)):
        %4 : NoneType = prim::Constant()
        %3 : int[] = prim::Constant[value=[4, 4]]()
        %u : Float(1, 1, 4, 4) = aten::upsample_nearest2d(%x, %3, %4)
        return (%u))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y_expected = at::upsample_nearest2d(x, {4, 4});

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

at::Tensor quantized_cat(
    c10::List<at::Tensor> const& xs,
    int64_t dim,
    double scale,
    int64_t zero) {
  const auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("quantized::cat", "")
                      .typed<at::Tensor(
                          c10::List<at::Tensor> const&,
                          int64_t,
                          c10::optional<double>,
                          c10::optional<int64_t>)>();
  return op.redispatch(
      DispatchKeySet({DispatchKey::QuantizedCPU}), xs, dim, scale, zero);
}

TEST_F(Quantization, QuantCatDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu), %y : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu), %z : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu)):
        %qdt : int = prim::Constant[value=13]()
        %qxz : int = prim::Constant[value=13]()
        %qxs : float = prim::Constant[value=0.1]()
        %qyz : int = prim::Constant[value=16]()
        %qys : float = prim::Constant[value=0.15]()
        %qzz : int = prim::Constant[value=19]()
        %qzs : float = prim::Constant[value=0.2]()
        %qx : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%x, %qxs, %qxz, %qdt)
        %qy : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%y, %qys, %qyz, %qdt)
        %qz : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%z, %qzs, %qzz, %qdt)
        %catx : Tensor[] = prim::ListConstruct(%qx, %qy, %qz)
        %catd : int = prim::Constant[value=0]()
        %qcat : QUInt8(3, 1, 2, 2) = quantized::cat(%catx, %catd, %qxs, %qxz)
        %cat : Float(3, 1, 2, 2) = aten::dequantize(%qcat)
        return (%cat))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto y = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto z = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto qx = at::quantize_per_tensor(x, 0.1f, 13, at::kQUInt8);
  auto qy = at::quantize_per_tensor(y, 0.15f, 16, at::kQUInt8);
  auto qz = at::quantize_per_tensor(z, 0.2f, 19, at::kQUInt8);
  auto qcat = quantized_cat({qx, qy, qz}, 0, 0.1f, 13);
  auto expected = at::dequantize(qcat);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x, y, z};
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  auto result = stack[0].toTensor();
  bool check = at::allclose(expected, result);
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y:\n" << y << std::endl;
    std::cout << "z:\n" << z << std::endl;
    std::cout << "qx:\n" << qx << std::endl;
    std::cout << "qy:\n" << qy << std::endl;
    std::cout << "qz:\n" << qz << std::endl;
    std::cout << "qcat:\n" << qcat << std::endl;
    std::cout << "expected:\n" << expected << std::endl;
    std::cout << "result:\n" << result << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}

} // namespace jit
} // namespace torch
