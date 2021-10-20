#pragma once

// This header can depend on ops/ and ts_backend/TsNode.h, as well as system/c++,
// PT,... but not on other lazy tensor core headers.

#include <memory>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/generic.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

inline NodePtr ScalarOp(const at::Scalar& value, lazy_tensors::Shape shape) {
  return torch::lazy::MakeNode<Scalar>(value, std::move(shape));
}
inline NodePtr ScalarOp(const at::Scalar& value,
                        lazy_tensors::PrimitiveType type) {
  return torch::lazy::MakeNode<Scalar>(value, type);
}

inline NodePtr ConstantOp(lazy_tensors::Literal value) {
  return torch::lazy::MakeNode<Constant>(std::move(value));
}

inline NodePtr GenericOp(OpKind op, OpList operands,
                         lazy_tensors::Shape shape, size_t num_outputs = 1,
                         torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           num_outputs, hash_seed);
}

inline NodePtr GenericOp(OpKind op, OpList operands,
                         const std::function<lazy_tensors::Shape()>& shape_fn,
                         size_t num_outputs = 1,
                         torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, shape_fn, num_outputs,
                           hash_seed);
}

inline NodePtr GenericOp(OpKind op, OpList operands,
                         size_t num_outputs = 1,
                         torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, num_outputs, hash_seed);
}

inline NodePtr GenericOp(OpKind op, lazy_tensors::Shape shape,
                         size_t num_outputs, torch::lazy::hash_t hash_seed) {
  return torch::lazy::MakeNode<Generic>(std::move(op), std::move(shape), num_outputs,
                           hash_seed);
}

NodePtr Acos(const torch::lazy::Value& input);

NodePtr Acosh(const torch::lazy::Value& input);

NodePtr Cos(const torch::lazy::Value& input);

NodePtr Cosh(const torch::lazy::Value& input);

NodePtr Asin(const torch::lazy::Value& input);

NodePtr Asinh(const torch::lazy::Value& input);

NodePtr Sin(const torch::lazy::Value& input);

NodePtr Sinh(const torch::lazy::Value& input);

NodePtr Atan(const torch::lazy::Value& input);

NodePtr Atanh(const torch::lazy::Value& input);

NodePtr Atan2(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Tan(const torch::lazy::Value& input);

NodePtr Tanh(const torch::lazy::Value& input);

NodePtr Neg(const torch::lazy::Value& input);

NodePtr SignOp(const torch::lazy::Value& input);

NodePtr Abs(const torch::lazy::Value& input);

NodePtr ReluOp(const torch::lazy::Value& input);

NodePtr Min(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Max(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Exp(const torch::lazy::Value& input);

NodePtr Expm1(const torch::lazy::Value& input);

NodePtr Erf(const torch::lazy::Value& input);

NodePtr Erfc(const torch::lazy::Value& input);

NodePtr Erfinv(const torch::lazy::Value& input);

NodePtr Log(const torch::lazy::Value& input);

NodePtr Log1p(const torch::lazy::Value& input);

NodePtr Sqrt(const torch::lazy::Value& input);

NodePtr Rsqrt(const torch::lazy::Value& input);

NodePtr ReciprocalOp(const torch::lazy::Value& input);

NodePtr Pow(const torch::lazy::Value& input, const torch::lazy::Value& exponent);

NodePtr Fmod(const torch::lazy::Value& dividend, const torch::lazy::Value& divisor);

NodePtr Not(const torch::lazy::Value& input);

NodePtr HardSigmoid(const torch::lazy::Value& input);

NodePtr HardSigmoidBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

std::tuple<NodePtr, NodePtr> LogSigmoid(const torch::lazy::Value& input);

NodePtr LogSigmoidBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                           const torch::lazy::Value& buffer);

NodePtr Sigmoid(const torch::lazy::Value& input);

NodePtr SiLU(const torch::lazy::Value& input);

NodePtr SigmoidBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& output);

NodePtr LogSoftmaxBackwardOp(const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
                             lazy_tensors::int64 dim);

NodePtr TSLogSoftmaxBackwardOp(const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
                               lazy_tensors::int64 dim, const torch::lazy::Value& self);

NodePtr SoftmaxBackwardOp(const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
                          lazy_tensors::int64 dim);

NodePtr TSSoftmaxBackwardOp(const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
                            lazy_tensors::int64 dim, const torch::lazy::Value& self);

NodePtr Clamp(const torch::lazy::Value& input, const torch::lazy::Value& min, const torch::lazy::Value& max);

NodePtr Ceil(const torch::lazy::Value& input);

NodePtr Floor(const torch::lazy::Value& input);

NodePtr Round(const torch::lazy::Value& input);

NodePtr Trunc(const torch::lazy::Value& input);

NodePtr FracOp(const torch::lazy::Value& input);

NodePtr Ger(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr AddMatMulOp(const torch::lazy::Value& input, const torch::lazy::Value& weight, const torch::lazy::Value& bias);

NodePtr MatMul(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs);

NodePtr AdaptiveAvgPool2dBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

NodePtr AdaptiveAvgPool3dBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

NodePtr ComparisonOp(c10::Symbol kind, const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Where(const torch::lazy::Value& condition, const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
               const at::Scalar& step, at::ScalarType scalar_type);

NodePtr BroadcastTensors(OpList tensors);

NodePtr Identity(lazy_tensors::int64 lines, lazy_tensors::int64 cols,
                 lazy_tensors::PrimitiveType element_type);

NodePtr Elu(const torch::lazy::Value& input, const at::Scalar& alpha,
            const at::Scalar& scale, const at::Scalar& input_scale);

NodePtr EluBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
                    const at::Scalar& alpha, const at::Scalar& scale,
                    const at::Scalar& input_scale);

NodePtr Lshift(const torch::lazy::Value& input, const at::Scalar& other);

NodePtr Lshift(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Rshift(const torch::lazy::Value& input, const at::Scalar& other);

NodePtr Rshift(const torch::lazy::Value& input, const torch::lazy::Value& other);

NodePtr Remainder(const torch::lazy::Value& input, const torch::lazy::Value& divisor);

NodePtr MaxUnary(const torch::lazy::Value& input);

NodePtr MinUnary(const torch::lazy::Value& input);

NodePtr Take(const torch::lazy::Value& input, const torch::lazy::Value& index);

NodePtr LogDet(const torch::lazy::Value& input);

NodePtr Inverse(const torch::lazy::Value& input);

NodePtr IsNan(const torch::lazy::Value& input);

NodePtr BaddBmm(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs, const torch::lazy::Value& bias,
                const torch::lazy::Value& product_multiplier, const torch::lazy::Value& bias_multiplier);

NodePtr Lerp(const torch::lazy::Value& start, const torch::lazy::Value& end, const torch::lazy::Value& weight);

NodePtr LogicalAnd(const torch::lazy::Value& input, const torch::lazy::Value& other);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
