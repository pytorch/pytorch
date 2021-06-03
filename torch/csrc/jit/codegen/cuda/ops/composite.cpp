#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ops/composite.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

ForwardDropoutResult dropout(TensorView* x, Val* prob) {
  auto p1m = sub(new Double(1.), prob);
  auto zero_check = add(eq(p1m, new Double(0.)), p1m);
  auto scale = div(new Double(1.), zero_check);
  return dropout(x, p1m, scale);
}

ForwardDropoutResult dropout(TensorView* x, Val* prob, Val* scale) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      prob != nullptr && prob->getDataType().has_value() &&
          prob->getDataType().value() == DataType::Double,
      "Probability is not a valid Double.");
  TORCH_INTERNAL_ASSERT(
      scale != nullptr && scale->getDataType().has_value() &&
          scale->getDataType().value() == DataType::Double,
      "Scale is not a valid Double.");

  auto rand_vals = unaryOp(UnaryOpType::RandLike, x);
  auto mask = lt(rand_vals, prob);
  auto apply_mask = mul(x, mask);
  auto y = mul(apply_mask, scale);

  return {y, mask};
}

TensorView* dropout_backward(TensorView* dy, TensorView* mask, Val* scale) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(mask != nullptr, "Mask is invalid");
  TORCH_INTERNAL_ASSERT(
      scale != nullptr && scale->getDataType().has_value() &&
          scale->getDataType().value() == DataType::Double,
      "Scale is not a valid Double.");

  auto grad_mask = mul(dy, mask);
  auto dx = mul(grad_mask, scale);

  return dx;
}

Val* softplus(Val* x, Val* beta, Val* threshold) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(beta != nullptr, "Beta is invalid.");
  TORCH_INTERNAL_ASSERT(
      threshold != nullptr, "Threshold is not a valid Double.");

  auto op_beta = mul(x, beta);
  auto maybe_result = div(
      unaryOp(UnaryOpType::Log1p, unaryOp(UnaryOpType::Exp, op_beta)), beta);
  auto y = where(gt(op_beta, threshold), x, maybe_result);
  return y;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
