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
  CHECK_EQ(check, 1);
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
  CHECK_EQ(check, 1);
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
  CHECK_EQ(check, 1);
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
  CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantUpsampleNearst2dDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : NoneType = prim::Constant()
        %4 : float[] = prim::Constant[value=[2., 2.]]()
        %qz : int = prim::Constant[value=13]()
        %qs : float = prim::Constant[value=0.1]()
        %q : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%x, %qs, %qz, %2)
        %qu : QUInt8(1, 1, 4, 4) = aten::upsample_nearest2d(%q, %3, %4)
        %6 : Float(1, 1, 4, 4) = aten::dequantize(%qu)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 13, at::kQUInt8);
  auto qu =
      at::upsample_nearest2d(q, c10::nullopt, at::ArrayRef<double>({2.f, 2.f}));
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
  CHECK_EQ(check, 1);
}

c10::intrusive_ptr<ConvPackedParamsBase<2>> quantized_conv2d_prepack(
    at::Tensor qweight,
    c10::optional<at::Tensor> bias,
    c10::List<int64_t> stride,
    c10::List<int64_t> padding,
    c10::List<int64_t> dilation,
    int64_t groups) {
  auto qconv2d_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::conv2d_prepack", "")
          .typed<c10::intrusive_ptr<ConvPackedParamsBase<2>>(
              at::Tensor,
              c10::optional<at::Tensor>,
              c10::List<int64_t>,
              c10::List<int64_t>,
              c10::List<int64_t>,
              int64_t)>();
  return qconv2d_prepack_op.call(
      qweight, bias, stride, padding, dilation, groups);
}

at::Tensor quantized_conv2d(
    at::Tensor qx,
    c10::intrusive_ptr<ConvPackedParamsBase<2>> packed_weight,
    double scale,
    int64_t zero) {
  auto qconv2d_op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("_quantized::conv2d", "")
                        .typed<at::Tensor(
                            at::Tensor,
                            const c10::intrusive_ptr<ConvPackedParamsBase<2>>&,
                            double,
                            int64_t)>();
  return qconv2d_op.call(qx, packed_weight, scale, zero);
}

at::Tensor quantized_conv2d_relu(
    at::Tensor qx,
    c10::intrusive_ptr<ConvPackedParamsBase<2>> packed_weight,
    double scale,
    int64_t zero) {
  auto qconv2d_op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("_quantized::conv2d_relu", "")
                        .typed<at::Tensor(
                            at::Tensor,
                            c10::intrusive_ptr<ConvPackedParamsBase<2>>,
                            double,
                            int64_t)>();
  return qconv2d_op.call(qx, packed_weight, scale, zero);
}

TEST_F(Quantization, QuantConv2dDequantUInt8) {
  const auto graph_string = R"IR(
      graph(%x : Float(1, 3, 2, 2, strides=[12, 4, 2, 1], device=cpu), %w : Float(2, 3, 2, 2, strides=[12, 4, 2, 1], device=cpu), %b : Float(2, strides=[1], device=cpu)):
        %qdti : int = prim::Constant[value=12]()
        %qdtui : int = prim::Constant[value=13]()
        %qxz : int = prim::Constant[value=130]()
        %qxs : float = prim::Constant[value=0.1]()
        %qwz : int = prim::Constant[value=13]()
        %qws : float = prim::Constant[value=0.1]()
        %qcz : int = prim::Constant[value=130]()
        %qcs : float = prim::Constant[value=0.1]()
        %s : int[] = prim::Constant[value=[1, 1]]()
        %p : int[] = prim::Constant[value=[0, 0]]()
        %d : int[] = prim::Constant[value=[1, 1]]()
        %g : int = prim::Constant[value=1]()
        %qw : QInt8(2, 3, 2, 2) = aten::quantize_per_tensor(%w, %qws, %qwz, %qdti)
        %qcp : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv2d_prepack(%qw, %b, %s, %p, %d, %g)
        %qx : QUInt8(1, 3, 2, 2) = aten::quantize_per_tensor(%x, %qxs, %qxz, %qdtui)
        %qc : QUInt8(1, 2, 1, 1) = quantized::conv2d(%qx, %qcp, %qcs, %qcz)
        %6 : Float(1, 2, 1, 1) = aten::dequantize(%qc)
        return (%6))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto x = at::rand({1, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto w = at::rand({2, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto q = at::quantize_per_tensor(x, 0.1f, 130, at::kQUInt8);
  auto qw = at::quantize_per_tensor(w, 0.1f, 13, at::kQInt8);
  auto qcp = quantized_conv2d_prepack(qw, b, {1, 1}, {0, 0}, {1, 1}, 1);
  auto qc = quantized_conv2d(q, qcp, 0.2f, 14);
  auto y_expected = at::dequantize(qc);

  TensorExprKernel k(graph);
  StmtPtr s = k.getCodeGenStmt();

  std::vector<IValue> stack = {IValue(x), IValue(w), IValue(b)};
  k.run(stack);
  auto y = stack[0].toTensor();
  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  CHECK_EQ(check, 1);
}

} // namespace jit
} // namespace torch
