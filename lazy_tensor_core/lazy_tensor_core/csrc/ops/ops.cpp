#include "lazy_tensor_core/csrc/ops/ops.h"

#include <c10/util/Half.h>

#include <cmath>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
using torch::lazy::ScopePusher;

#define PTLTC_UNARY_OP(name, sym)                          \
  NodePtr name(const torch::lazy::Value& input) {                       \
    return GenericOp(OpKind(sym), {input}, ir::GetShapeFromTsValue(input)); \
  }

#define PTLTC_BINARY_OP(name, sym)                                           \
  NodePtr name(const torch::lazy::Value& input0, const torch::lazy::Value& input1) {                   \
    NodePtr node = GenericOp(OpKind(sym), {input0, input1});                 \
    std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(                                                  \
        [&]() { return compiler::InferShape(node.get()); }); \
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
PTLTC_UNARY_OP(Neg, at::aten::neg);
PTLTC_UNARY_OP(Expm1, at::aten::expm1);
PTLTC_UNARY_OP(Log, at::aten::log);
PTLTC_UNARY_OP(Log1p, at::aten::log1p);
PTLTC_UNARY_OP(Erf, at::aten::erf);
PTLTC_UNARY_OP(Erfc, at::aten::erfc);
PTLTC_UNARY_OP(Erfinv, at::aten::erfinv);
PTLTC_UNARY_OP(Rsqrt, at::aten::rsqrt);
PTLTC_UNARY_OP(Ceil, at::aten::ceil);
PTLTC_UNARY_OP(Round, at::aten::round);
PTLTC_UNARY_OP(Not, at::aten::bitwise_not);
PTLTC_UNARY_OP(IsNan, at::aten::isnan);

PTLTC_BINARY_OP(Min, at::aten::min);
PTLTC_BINARY_OP(Max, at::aten::max);
PTLTC_BINARY_OP(Pow, at::aten::pow);
PTLTC_BINARY_OP(Fmod, at::aten::fmod);
PTLTC_BINARY_OP(Atan2, at::aten::atan2);

NodePtr ReciprocalOp(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::reciprocal), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr SignOp(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::sign), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr Abs(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::abs), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr HardSigmoid(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::hardsigmoid), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr HardSigmoidBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::hardsigmoid_backward), {grad_output, input},
                   ir::GetShapeFromTsValue(input));
}

NodePtr SiLU(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::silu), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr Sigmoid(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::sigmoid), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr SigmoidBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& output) {
  return grad_output * (ScalarOp(1, ir::GetShapeFromTsValue(output)) - output) * output;
}

NodePtr Clamp(const torch::lazy::Value& input, const torch::lazy::Value& min, const torch::lazy::Value& max) {
  return GenericOp(OpKind(at::aten::clamp), {input, min, max}, ir::GetShapeFromTsValue(input));
}

NodePtr Ger(const torch::lazy::Value& input, const torch::lazy::Value& other) {
  NodePtr node = GenericOp(OpKind(at::aten::ger), {input, other});
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

NodePtr AdaptiveAvgPool3dBackward(const torch::lazy::Value& grad_output,
                                  const torch::lazy::Value& input) {
  NodePtr node = GenericOp(OpKind(at::aten::adaptive_avg_pool3d_backward),
                           {grad_output, input});
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

NodePtr AdaptiveAvgPool2dBackward(const torch::lazy::Value& grad_output,
                                  const torch::lazy::Value& input) {
  NodePtr node = GenericOp(OpKind(at::aten::adaptive_avg_pool2d_backward),
                           {grad_output, input});
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

NodePtr ComparisonOp(c10::Symbol kind, const torch::lazy::Value& input, const torch::lazy::Value& other) {
  NodePtr node = GenericOp(OpKind(kind), {input, other});
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

NodePtr Where(const torch::lazy::Value& condition, const torch::lazy::Value& input, const torch::lazy::Value& other) {
  return GenericOp(OpKind(at::aten::where), {condition, input, other},
                   ir::GetShapeFromTsValue(input));
}

NodePtr BroadcastTensors(OpList tensors) {
  NodePtr node = GenericOp(OpKind(at::aten::broadcast_tensors), tensors,
                           /*num_outputs=*/tensors.size());
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

NodePtr Identity(int64_t lines, int64_t cols, c10::ScalarType element_type) {
  return GenericOp(
      OpKind(at::aten::eye),
      lazy_tensors::ShapeUtil::MakeShape(element_type, {lines, cols}),
      /*num_outputs=*/1, torch::lazy::MHash(lines, cols));
}

NodePtr Lshift(const torch::lazy::Value& input, const at::Scalar& other) {
  ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * ScalarOp(pow(2, other.to<double>()), ir::GetShapeFromTsValue(input));
}

NodePtr Lshift(const torch::lazy::Value& input, const torch::lazy::Value& other) {
  ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * Pow(ScalarOp(2, ir::GetShapeFromTsValue(input)), other);
}

NodePtr Rshift(const torch::lazy::Value& input, const at::Scalar& other) {
  ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / ScalarOp(pow(2, other.to<double>()), ir::GetShapeFromTsValue(input));
}

NodePtr Rshift(const torch::lazy::Value& input, const torch::lazy::Value& other) {
  ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / Pow(ScalarOp(2, ir::GetShapeFromTsValue(input)), other);
}

NodePtr Remainder(const torch::lazy::Value& input, const torch::lazy::Value& divisor) {
  ScopePusher ir_scope(at::aten::remainder.toQualString());
  NodePtr f = Fmod(input, Abs(divisor));
  return f + divisor * ComparisonOp(at::aten::lt, SignOp(f) * SignOp(divisor),
                                    ScalarOp(0, ir::GetShapeFromTsValue(input)));
}

NodePtr MaxUnary(const torch::lazy::Value& input) {
  CHECK_GT(lazy_tensors::ShapeUtil::ElementsIn(ir::GetShapeFromTsValue(input)),
           0);
  return GenericOp(
      OpKind(at::aten::max), {input},
      lazy_tensors::ShapeUtil::MakeShape(ir::GetShapeFromTsValue(input).scalar_type(), {}));
}

NodePtr MinUnary(const torch::lazy::Value& input) {
  CHECK_GT(lazy_tensors::ShapeUtil::ElementsIn(ir::GetShapeFromTsValue(input)),
           0);
  return GenericOp(
      OpKind(at::aten::min), {input},
      lazy_tensors::ShapeUtil::MakeShape(ir::GetShapeFromTsValue(input).scalar_type(), {}));
}

NodePtr Take(const torch::lazy::Value& input, const torch::lazy::Value& index) {
  torch::lazy::Shape result_shape = ir::GetShapeFromTsValue(index);
  result_shape.set_scalar_type(ir::GetShapeFromTsValue(input).scalar_type());
  return GenericOp(OpKind(at::aten::take), {input, index},
                   std::move(result_shape));
}

NodePtr Inverse(const torch::lazy::Value& input) {
  return GenericOp(OpKind(at::aten::inverse), {input}, ir::GetShapeFromTsValue(input));
}

NodePtr Lerp(const torch::lazy::Value& start, const torch::lazy::Value& end, const torch::lazy::Value& weight) {
  ScopePusher ir_scope(at::aten::lerp.toQualString());
  return start + weight * (end - start);
}

NodePtr LogicalAnd(const torch::lazy::Value& input, const torch::lazy::Value& other) {
  NodePtr node = GenericOp(OpKind(at::aten::logical_and), {input, other});
  std::dynamic_pointer_cast<TsNode>(node)->SetShapeDeferred(
      [&]() { return compiler::InferShape(node.get()); });
  return node;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
