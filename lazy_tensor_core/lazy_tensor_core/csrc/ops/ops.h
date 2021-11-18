#pragma once

// This header can depend on ops/ and ts_backend/torch::lazy::TsNode.h, as well
// as system/c++, PT,... but not on other lazy tensor core headers.

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <memory>

#include "lazy_tensor_core/csrc/ops/generic.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     torch::lazy::Shape shape) {
  return torch::lazy::MakeNode<Scalar>(value, std::move(shape));
}
inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     c10::ScalarType type) {
  return torch::lazy::MakeNode<Scalar>(value, type);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op, torch::lazy::OpList operands,
                         torch::lazy::Shape shape, size_t num_outputs = 1,
                         torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, torch::lazy::OpList operands,
    const std::function<torch::lazy::Shape()>& shape_fn, size_t num_outputs = 1,
    torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, shape_fn, num_outputs,
                           hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, torch::lazy::OpList operands,
    size_t num_outputs = 1,
    torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op,
                                      torch::lazy::Shape shape,
                                      size_t num_outputs,
                                      torch::lazy::hash_t hash_seed) {
  return torch::lazy::MakeNode<Generic>(std::move(op), std::move(shape), num_outputs,
                           hash_seed);
}

torch::lazy::NodePtr Acos(const torch::lazy::Value& input);

torch::lazy::NodePtr Acosh(const torch::lazy::Value& input);

torch::lazy::NodePtr Cos(const torch::lazy::Value& input);

torch::lazy::NodePtr Cosh(const torch::lazy::Value& input);

torch::lazy::NodePtr Asin(const torch::lazy::Value& input);

torch::lazy::NodePtr Asinh(const torch::lazy::Value& input);

torch::lazy::NodePtr Sin(const torch::lazy::Value& input);

torch::lazy::NodePtr Sinh(const torch::lazy::Value& input);

torch::lazy::NodePtr Atan(const torch::lazy::Value& input);

torch::lazy::NodePtr Atanh(const torch::lazy::Value& input);

torch::lazy::NodePtr Atan2(const torch::lazy::Value& input,
                           const torch::lazy::Value& other);

torch::lazy::NodePtr Tan(const torch::lazy::Value& input);

torch::lazy::NodePtr Neg(const torch::lazy::Value& input);

torch::lazy::NodePtr SignOp(const torch::lazy::Value& input);

torch::lazy::NodePtr Abs(const torch::lazy::Value& input);

torch::lazy::NodePtr Min(const torch::lazy::Value& input,
                         const torch::lazy::Value& other);

torch::lazy::NodePtr Max(const torch::lazy::Value& input,
                         const torch::lazy::Value& other);

torch::lazy::NodePtr Exp(const torch::lazy::Value& input);

torch::lazy::NodePtr Expm1(const torch::lazy::Value& input);

torch::lazy::NodePtr Erf(const torch::lazy::Value& input);

torch::lazy::NodePtr Erfc(const torch::lazy::Value& input);

torch::lazy::NodePtr Erfinv(const torch::lazy::Value& input);

torch::lazy::NodePtr Log(const torch::lazy::Value& input);

torch::lazy::NodePtr Log1p(const torch::lazy::Value& input);

torch::lazy::NodePtr Rsqrt(const torch::lazy::Value& input);

torch::lazy::NodePtr ReciprocalOp(const torch::lazy::Value& input);

torch::lazy::NodePtr Pow(const torch::lazy::Value& input,
                         const torch::lazy::Value& exponent);

torch::lazy::NodePtr Fmod(const torch::lazy::Value& dividend,
                          const torch::lazy::Value& divisor);

torch::lazy::NodePtr Not(const torch::lazy::Value& input);

torch::lazy::NodePtr HardSigmoid(const torch::lazy::Value& input);

torch::lazy::NodePtr HardSigmoidBackward(const torch::lazy::Value& grad_output,
                                         const torch::lazy::Value& input);

torch::lazy::NodePtr Sigmoid(const torch::lazy::Value& input);

torch::lazy::NodePtr SiLU(const torch::lazy::Value& input);

torch::lazy::NodePtr SigmoidBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& output);

torch::lazy::NodePtr LogSoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                          const torch::lazy::Value& output,
                                          int64_t dim);

torch::lazy::NodePtr TSLogSoftmaxBackwardOp(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& output,
    int64_t dim, const torch::lazy::Value& self);

torch::lazy::NodePtr SoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                       const torch::lazy::Value& output,
                                       int64_t dim);

torch::lazy::NodePtr TSSoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                         const torch::lazy::Value& output,
                                         int64_t dim,
                                         const torch::lazy::Value& self);

torch::lazy::NodePtr Clamp(const torch::lazy::Value& input,
                           const torch::lazy::Value& min,
                           const torch::lazy::Value& max);

torch::lazy::NodePtr Ceil(const torch::lazy::Value& input);

torch::lazy::NodePtr Round(const torch::lazy::Value& input);

torch::lazy::NodePtr Ger(const torch::lazy::Value& input,
                         const torch::lazy::Value& other);

torch::lazy::NodePtr AdaptiveAvgPool2dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

torch::lazy::NodePtr AdaptiveAvgPool3dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input);

torch::lazy::NodePtr ComparisonOp(c10::Symbol kind,
                                  const torch::lazy::Value& input,
                                  const torch::lazy::Value& other);

torch::lazy::NodePtr Where(const torch::lazy::Value& condition,
                           const torch::lazy::Value& input,
                           const torch::lazy::Value& other);

torch::lazy::NodePtr BroadcastTensors(torch::lazy::OpList tensors);

torch::lazy::NodePtr Identity(int64_t lines, int64_t cols,
                              c10::ScalarType element_type);

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const at::Scalar& other);

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other);

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const at::Scalar& other);

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other);

torch::lazy::NodePtr Remainder(const torch::lazy::Value& input,
                               const torch::lazy::Value& divisor);

torch::lazy::NodePtr MaxUnary(const torch::lazy::Value& input);

torch::lazy::NodePtr MinUnary(const torch::lazy::Value& input);

torch::lazy::NodePtr Take(const torch::lazy::Value& input,
                          const torch::lazy::Value& index);

torch::lazy::NodePtr Inverse(const torch::lazy::Value& input);

torch::lazy::NodePtr IsNan(const torch::lazy::Value& input);

torch::lazy::NodePtr BaddBmm(const torch::lazy::Value& lhs,
                             const torch::lazy::Value& rhs,
                             const torch::lazy::Value& bias,
                             const torch::lazy::Value& product_multiplier,
                             const torch::lazy::Value& bias_multiplier);

torch::lazy::NodePtr Lerp(const torch::lazy::Value& start,
                          const torch::lazy::Value& end,
                          const torch::lazy::Value& weight);

torch::lazy::NodePtr LogicalAnd(const torch::lazy::Value& input,
                                const torch::lazy::Value& other);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
