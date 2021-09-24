#include "lazy_tensor_core/csrc/ops/ops.h"

#include <cmath>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/ts_log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_softmax_backward.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

#define PTLTC_UNARY_OP(name, sym)                          \
  NodePtr name(const Value& input) {                       \
    return GenericOp(OpKind(sym), {input}, input.shape()); \
  }

#define PTLTC_BINARY_OP(name, sym)                                           \
  NodePtr name(const Value& input0, const Value& input1) {                   \
    NodePtr node = GenericOp(OpKind(sym), {input0, input1});                 \
    node->SetShapeDeferred(                                                  \
        [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); }); \
    return node;                                                             \
  }

PTLTC_UNARY_OP(Acos, at::aten::acos);
PTLTC_UNARY_OP(Acosh, at::aten::acosh);
PTLTC_UNARY_OP(Cos, at::aten::cos);
PTLTC_UNARY_OP(Cosh, at::aten::cosh);
PTLTC_UNARY_OP(Asin, at::aten::asin);
PTLTC_UNARY_OP(Asinh, at::aten::asinh);
PTLTC_UNARY_OP(Sin, at::aten::sin);
PTLTC_UNARY_OP(Sinh, at::aten::sinh);
PTLTC_UNARY_OP(Atan, at::aten::atan);
PTLTC_UNARY_OP(Atanh, at::aten::atanh);
PTLTC_UNARY_OP(Tan, at::aten::tan);
PTLTC_UNARY_OP(Tanh, at::aten::tanh);
PTLTC_UNARY_OP(Neg, at::aten::neg);
PTLTC_UNARY_OP(Exp, at::aten::exp);
PTLTC_UNARY_OP(Expm1, at::aten::expm1);
PTLTC_UNARY_OP(Log, at::aten::log);
PTLTC_UNARY_OP(Log1p, at::aten::log1p);
PTLTC_UNARY_OP(Erf, at::aten::erf);
PTLTC_UNARY_OP(Erfc, at::aten::erfc);
PTLTC_UNARY_OP(Erfinv, at::aten::erfinv);
PTLTC_UNARY_OP(Sqrt, at::aten::sqrt);
PTLTC_UNARY_OP(Rsqrt, at::aten::rsqrt);
PTLTC_UNARY_OP(Ceil, at::aten::ceil);
PTLTC_UNARY_OP(Floor, at::aten::floor);
PTLTC_UNARY_OP(Round, at::aten::round);
PTLTC_UNARY_OP(Not, at::aten::bitwise_not);
PTLTC_UNARY_OP(IsNan, at::aten::isnan);

PTLTC_BINARY_OP(Min, at::aten::min);
PTLTC_BINARY_OP(Max, at::aten::max);
PTLTC_BINARY_OP(Pow, at::aten::pow);
PTLTC_BINARY_OP(Fmod, at::aten::fmod);
PTLTC_BINARY_OP(Atan2, at::aten::atan2);

NodePtr Trunc(const Value& input) { return Floor(Abs(input)) * SignOp(input); }

NodePtr FracOp(const Value& input) { return input - Trunc(input); }

NodePtr ReciprocalOp(const Value& input) {
  return GenericOp(OpKind(at::aten::reciprocal), {input}, input.shape());
}

NodePtr SignOp(const Value& input) {
  return GenericOp(OpKind(at::aten::sign), {input}, input.shape());
}

NodePtr Abs(const Value& input) {
  return GenericOp(OpKind(at::aten::abs), {input}, input.shape());
}

NodePtr ReluOp(const Value& input) {
  NodePtr node = GenericOp(OpKind(at::aten::relu), {input});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr HardSigmoid(const Value& input) {
  return GenericOp(OpKind(at::aten::hardsigmoid), {input}, input.shape());
}

NodePtr HardSigmoidBackward(const Value& grad_output, const Value& input) {
  return GenericOp(OpKind(at::aten::hardsigmoid_backward), {grad_output, input},
                   input.shape());
}

std::tuple<NodePtr, NodePtr> LogSigmoid(const Value& input) {
  ScopePusher ir_scope(at::aten::log_sigmoid.toQualString());
  // Use log-sum-exp trick to avoid overflow.
  NodePtr neg_input = Neg(input);
  NodePtr max_elem = Max(ScalarOp(0, input.shape()), neg_input);
  NodePtr buffer = Exp(Neg(max_elem)) + Exp(neg_input - max_elem);
  NodePtr output = Neg(max_elem + Log(buffer));
  return std::make_tuple(output, buffer);
}

NodePtr LogSigmoidBackward(const Value& grad_output, const Value& input,
                           const Value& buffer) {
  ScopePusher ir_scope(at::aten::log_sigmoid_backward.toQualString());
  NodePtr zero = ScalarOp(0, input.shape());
  NodePtr one = ScalarOp(1, input.shape());
  NodePtr minus_one = ScalarOp(-1, input.shape());
  NodePtr max_deriv =
      Where(ComparisonOp(at::aten::lt, input, zero), minus_one, zero);
  NodePtr sign = Where(ComparisonOp(at::aten::lt, input, zero), one, minus_one);
  return grad_output * (Neg(max_deriv) - sign * (buffer - one) / buffer);
}

NodePtr SiLU(const Value& input) {
  return GenericOp(OpKind(at::aten::silu), {input}, input.shape());
}

NodePtr Sigmoid(const Value& input) {
  return GenericOp(OpKind(at::aten::sigmoid), {input}, input.shape());
}

NodePtr SigmoidBackward(const Value& grad_output, const Value& output) {
  return grad_output * (ScalarOp(1, output.shape()) - output) * output;
}

NodePtr LogSoftmaxBackwardOp(const Value& grad_output, const Value& output,
                             lazy_tensors::int64 dim) {
  return MakeNode<LogSoftmaxBackward>(
      grad_output, output,
      Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()));
}

NodePtr TSLogSoftmaxBackwardOp(const Value& grad_output, const Value& output,
                               lazy_tensors::int64 dim, const Value& self) {
  return MakeNode<TSLogSoftmaxBackward>(
      grad_output, output,
      Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()),
      self);
}

NodePtr SoftmaxBackwardOp(const Value& grad_output, const Value& output,
                          lazy_tensors::int64 dim) {
  return MakeNode<SoftmaxBackward>(
      grad_output, output,
      Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()));
}

NodePtr TSSoftmaxBackwardOp(const Value& grad_output, const Value& output,
                            lazy_tensors::int64 dim, const Value& self) {
  return MakeNode<TSSoftmaxBackward>(
      grad_output, output,
      Helpers::GetCanonicalDimensionIndex(dim, grad_output.shape().rank()),
      self);
}

NodePtr Clamp(const Value& input, const Value& min, const Value& max) {
  return GenericOp(OpKind(at::aten::clamp), {input, min, max}, input.shape());
}

NodePtr Ger(const Value& input, const Value& other) {
  NodePtr node = GenericOp(OpKind(at::aten::ger), {input, other});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr AddMatMulOp(const Value& input, const Value& weight,
                    const Value& bias) {
  NodePtr node = GenericOp(OpKind(at::aten::addmm), {input, weight, bias});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr MatMul(const Value& lhs, const Value& rhs) {
  NodePtr node = GenericOp(OpKind(at::aten::matmul), {lhs, rhs});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr AdaptiveAvgPool3dBackward(const Value& grad_output,
                                  const Value& input) {
  NodePtr node = GenericOp(OpKind(at::aten::adaptive_avg_pool3d_backward),
                           {grad_output, input});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output,
                                  const Value& input) {
  NodePtr node = GenericOp(OpKind(at::aten::adaptive_avg_pool2d_backward),
                           {grad_output, input});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other) {
  NodePtr node = GenericOp(OpKind(kind), {input, other});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr Where(const Value& condition, const Value& input, const Value& other) {
  return GenericOp(OpKind(at::aten::where), {condition, input, other},
                   input.shape());
}

NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
               const at::Scalar& step, at::ScalarType scalar_type) {
  lazy_tensors::PrimitiveType type = MakeLtcPrimitiveType(scalar_type,
                                                          /*device=*/nullptr);
  LTC_CHECK_NE(step.toDouble(), 0.0);
  LTC_CHECK(!std::isnan(start.toDouble()) && !std::isnan(end.toDouble()))
      << "unsupported range: " << start.toDouble() << " -> " << end.toDouble();
  LTC_CHECK((start.toDouble() <= end.toDouble() && step.toDouble() > 0.0) ||
            (start.toDouble() >= end.toDouble() && step.toDouble() < 0.0));
  lazy_tensors::Literal values;
  switch (type) {
    case lazy_tensors::PrimitiveType::BF16:
      values = Helpers::Range<lazy_tensors::bfloat16>(
          static_cast<lazy_tensors::bfloat16>(start.toFloat()),
          static_cast<lazy_tensors::bfloat16>(end.toFloat()),
          static_cast<lazy_tensors::bfloat16>(step.toFloat()));
      break;
    case lazy_tensors::PrimitiveType::F16:
      values = Helpers::Range<lazy_tensors::half>(
          static_cast<lazy_tensors::half>(start.toHalf()),
          static_cast<lazy_tensors::half>(end.toHalf()),
          static_cast<lazy_tensors::half>(step.toHalf()));
      break;
    case lazy_tensors::PrimitiveType::F32:
      values =
          Helpers::Range<float>(start.toFloat(), end.toFloat(), step.toFloat());
      break;
    case lazy_tensors::PrimitiveType::F64:
      values = Helpers::Range<double>(start.toDouble(), end.toDouble(),
                                      step.toDouble());
      break;
    case lazy_tensors::PrimitiveType::U8:
      values = Helpers::Range<lazy_tensors::uint8>(start.toByte(), end.toByte(),
                                                   step.toByte());
      break;
    case lazy_tensors::PrimitiveType::S8:
      values = Helpers::Range<lazy_tensors::int8>(start.toChar(), end.toChar(),
                                                  step.toChar());
      break;
    case lazy_tensors::PrimitiveType::S16:
      values = Helpers::Range<lazy_tensors::int16>(
          start.toShort(), end.toShort(), step.toShort());
      break;
    case lazy_tensors::PrimitiveType::U16:
      values = Helpers::Range<lazy_tensors::uint16>(start.toInt(), end.toInt(),
                                                    step.toInt());
      break;
    case lazy_tensors::PrimitiveType::S32:
      values = Helpers::Range<lazy_tensors::int32>(start.toInt(), end.toInt(),
                                                   step.toInt());
      break;
    case lazy_tensors::PrimitiveType::U32:
      values = Helpers::Range<lazy_tensors::uint32>(
          start.toLong(), end.toLong(), step.toLong());
      break;
    case lazy_tensors::PrimitiveType::S64:
      values = Helpers::Range<lazy_tensors::int64>(start.toLong(), end.toLong(),
                                                   step.toLong());
      break;
    case lazy_tensors::PrimitiveType::U64:
      values = Helpers::Range<lazy_tensors::uint64>(
          start.toLong(), end.toLong(), step.toLong());
      break;
    default:
      LTC_ERROR() << "Type not supported: " << type;
  }
  return MakeNode<Constant>(std::move(values));
}

NodePtr BroadcastTensors(lazy_tensors::Span<const Value> tensors) {
  NodePtr node = GenericOp(OpKind(at::aten::broadcast_tensors), tensors,
                           /*num_outputs=*/tensors.size());
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr Norm(const Value& input, const c10::optional<at::Scalar>& p,
             c10::optional<at::ScalarType> dtype,
             lazy_tensors::Span<const lazy_tensors::int64> dims, bool keepdim) {
  ScopePusher ir_scope(at::aten::norm.toQualString());
  auto dimensions = lazy_tensors::util::ToVector<lazy_tensors::int64>(dims);
  if (dimensions.empty()) {
    dimensions =
        lazy_tensors::util::Iota<lazy_tensors::int64>(input.shape().rank());
  }
  if (!p.has_value() || p->toDouble() == 2.0) {
    NodePtr square = input * input;
    NodePtr result = MakeNode<Sum>(square, dimensions, keepdim, dtype);
    return Sqrt(result);
  }
  double norm_value = p->toDouble();
  if (norm_value == 1.0) {
    // Contrary to documentation, norm(p=1) has nothing to do with traces and
    // standard mathematical definitions of nuclear norms:
    //
    //   >>> import torch
    //   >>> x = torch.randn(4, 4)
    //   >>> print(torch.norm(x, 1))
    //   tensor(11.9437)
    //   >>> print(torch.trace(x.abs()))
    //   tensor(3.1235)
    //   >>> print(x.abs().sum())
    //   tensor(11.9437)
    return MakeNode<Sum>(Abs(input), dimensions, keepdim, dtype);
  }
  // Generic sum(x^p)^(1/p) norms.
  NodePtr norm_exp = ScalarOp(norm_value, input.shape().element_type());
  NodePtr norm_exp_inv =
      ScalarOp(1.0 / norm_value, input.shape().element_type());
  NodePtr exp = Pow(Abs(input), norm_exp);
  NodePtr result = MakeNode<Sum>(exp, dimensions, keepdim, dtype);
  return Pow(result, norm_exp_inv);
}

NodePtr Identity(lazy_tensors::int64 lines, lazy_tensors::int64 cols,
                 lazy_tensors::PrimitiveType element_type) {
  return GenericOp(
      OpKind(at::aten::eye),
      lazy_tensors::ShapeUtil::MakeShape(element_type, {lines, cols}),
      /*num_outputs=*/1, lazy_tensors::util::MHash(lines, cols));
}

NodePtr Elu(const Value& input, const at::Scalar& alpha,
            const at::Scalar& scale, const at::Scalar& input_scale) {
  ScopePusher ir_scope(at::aten::elu.toQualString());
  const lazy_tensors::Shape& shape = input.shape();
  NodePtr scaled_input = input * ScalarOp(input_scale, shape);
  NodePtr zero = ScalarOp(0, shape);
  NodePtr one = ScalarOp(1, shape);
  NodePtr alpha_scalar = ScalarOp(alpha, shape);
  return Where(ComparisonOp(at::aten::le, input, zero),
               alpha_scalar * (Exp(scaled_input) - one), input) *
         ScalarOp(scale, shape);
}

NodePtr EluBackward(const Value& grad_output, const Value& output,
                    const at::Scalar& alpha, const at::Scalar& scale,
                    const at::Scalar& input_scale) {
  ScopePusher ir_scope(at::aten::elu_backward.toQualString());
  const lazy_tensors::Shape& shape = grad_output.shape();
  NodePtr negative_output_branch =
      ScalarOp(input_scale, shape) *
      (output + ScalarOp(alpha, shape) * ScalarOp(scale, shape));
  NodePtr positive_output_branch = ScalarOp(scale, shape);
  return grad_output *
         Where(ComparisonOp(at::aten::gt, output, ScalarOp(0, shape)),
               positive_output_branch, negative_output_branch);
}

NodePtr Gelu(const Value& input) {
  ScopePusher ir_scope("aten::gelu");
  // input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
  const lazy_tensors::Shape& shape = input.shape();
  return input * ScalarOp(0.5, shape) *
         (Erf(input * ScalarOp(M_SQRT1_2, shape)) + ScalarOp(1.0, shape));
}

NodePtr GeluBackward(const Value& grad, const Value& input) {
  ScopePusher ir_scope("aten::gelu_backward");
  const float kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  const lazy_tensors::Shape& shape = input.shape();
  NodePtr scratch = Erf(input * ScalarOp(M_SQRT1_2, shape));
  NodePtr dinput = Exp(input * input * ScalarOp(-0.5, shape));
  return grad * (ScalarOp(0.5, shape) * (ScalarOp(1.0, shape) + scratch) +
                 input * dinput * ScalarOp(kAlpha, shape));
}

NodePtr Lshift(const Value& input, const at::Scalar& other) {
  ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * ScalarOp(pow(2, other.to<double>()), input.shape());
}

NodePtr Lshift(const Value& input, const Value& other) {
  ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * Pow(ScalarOp(2, input.shape()), other);
}

NodePtr Rshift(const Value& input, const at::Scalar& other) {
  ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / ScalarOp(pow(2, other.to<double>()), input.shape());
}

NodePtr Rshift(const Value& input, const Value& other) {
  ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / Pow(ScalarOp(2, input.shape()), other);
}

NodePtr Remainder(const Value& input, const Value& divisor) {
  ScopePusher ir_scope(at::aten::remainder.toQualString());
  NodePtr f = Fmod(input, Abs(divisor));
  return f + divisor * ComparisonOp(at::aten::lt, SignOp(f) * SignOp(divisor),
                                    ScalarOp(0, input.shape()));
}

NodePtr MaxUnary(const Value& input) {
  LTC_CHECK_GT(lazy_tensors::ShapeUtil::ElementsIn(input.shape()), 0);
  return GenericOp(
      OpKind(at::aten::max), {input},
      lazy_tensors::ShapeUtil::MakeShape(input.shape().element_type(), {}));
}

NodePtr MinUnary(const Value& input) {
  LTC_CHECK_GT(lazy_tensors::ShapeUtil::ElementsIn(input.shape()), 0);
  return GenericOp(
      OpKind(at::aten::min), {input},
      lazy_tensors::ShapeUtil::MakeShape(input.shape().element_type(), {}));
}

NodePtr Take(const Value& input, const Value& index) {
  lazy_tensors::Shape result_shape = index.shape();
  result_shape.set_element_type(input.shape().element_type());
  return GenericOp(OpKind(at::aten::take), {input, index},
                   std::move(result_shape));
}

NodePtr LogDet(const Value& input) {
  const lazy_tensors::Shape& input_shape = input.shape();
  LTC_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,N,N
  lazy_tensors::Shape logdet_shape(input_shape);
  logdet_shape.DeleteDimension(input_shape.rank() - 1);
  logdet_shape.DeleteDimension(input_shape.rank() - 2);
  return GenericOp(OpKind(at::aten::logdet), {input}, logdet_shape);
}

NodePtr Inverse(const Value& input) {
  return GenericOp(OpKind(at::aten::inverse), {input}, input.shape());
}

NodePtr BaddBmm(const Value& lhs, const Value& rhs, const Value& bias,
                const Value& product_multiplier, const Value& bias_multiplier) {
  NodePtr node =
      GenericOp(OpKind(at::aten::baddbmm),
                {lhs, rhs, bias, product_multiplier, bias_multiplier});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

NodePtr Lerp(const Value& start, const Value& end, const Value& weight) {
  ScopePusher ir_scope(at::aten::lerp.toQualString());
  return start + weight * (end - start);
}

NodePtr LogicalAnd(const Value& input, const Value& other) {
  NodePtr node = GenericOp(OpKind(at::aten::logical_and), {input, other});
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
