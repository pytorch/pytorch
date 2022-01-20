#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ops/composite.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>

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

  auto rand_vals = randlike(x);
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
  auto maybe_result = div(log1p(exp(op_beta)), beta);
  auto y = where(gt(op_beta, threshold), x, maybe_result);
  return y;
}

LstmResult lstm(
    TensorView* prev_cell,
    TensorView* in_x,
    TensorView* forget_x,
    TensorView* cell_x,
    TensorView* out_x) {
  TORCH_INTERNAL_ASSERT(
      prev_cell != nullptr, "Previous cell state is invalid.");
  TORCH_INTERNAL_ASSERT(in_x != nullptr, "In-gate input is invalid");
  TORCH_INTERNAL_ASSERT(forget_x != nullptr, "Forget-gate input is invalid");
  TORCH_INTERNAL_ASSERT(cell_x != nullptr, "Cell-gate input is invalid");
  TORCH_INTERNAL_ASSERT(out_x != nullptr, "Out-gate input is invalid");

  const auto in_gate = sigmoid(in_x);
  const auto forget_gate = sigmoid(forget_x);
  const auto cell_gate = tanh(cell_x);
  const auto out_gate = sigmoid(out_x);

  const auto cell = add(mul(forget_gate, prev_cell), mul(in_gate, cell_gate));
  const auto hidden = mul(out_gate, tanh(cell));

  return {cell, hidden};
}

Val* fast_gelu(Val* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_cube = mul(x, mul(x, x));

  auto inner_1 = mul(new Double(kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(new Double(kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto out = mul(x, add(new Double(1.), tanh_inner));
  auto y = mul(new Double(0.5), out);
  return y;
}

Val* fast_gelu_backward(Val* dy, Val* x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_sq = mul(x, x);
  auto x_cube = mul(x, x_sq);

  auto inner_1 = mul(new Double(kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(new Double(kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto left = mul(new Double(0.5), x);
  auto right = add(new Double(1.), tanh_inner);

  auto left_derivative = mul(new Double(0.5), right);

  auto tanh_inner_sq = mul(tanh_inner, tanh_inner);
  auto tanh_derivative = sub(new Double(1), tanh_inner_sq);

  auto constant_mul_x_sq = mul(new Double(kBeta * 3 * kKappa), x_sq);
  auto inner_derivative = add(new Double(kBeta), constant_mul_x_sq);
  auto right_derivative = mul(left, mul(tanh_derivative, inner_derivative));

  auto dx = mul(dy, add(left_derivative, right_derivative));
  return dx;
}

Val* gelu_backward(Val* dy, Val* x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  const double kHalf = 0.5;

  auto cdf_1 = mul(x, new Double(M_SQRT1_2));
  auto cdf_2 = erf(cdf_1);
  auto cdf_3 = add(cdf_2, new Double(1.));
  auto cdf_4 = mul(cdf_3, new Double(kHalf));

  auto pdf_1 = mul(x, x);
  auto pdf_2 = mul(pdf_1, new Double(-kHalf));
  auto pdf_3 = exp(pdf_2);

  auto out = addcmul(cdf_4, x, pdf_3, new Double(kAlpha));
  auto dx = mul(out, dy);
  return dx;
}

namespace {

//! Transform TensorView according to keep, merge, and split transformations.
//! Trivial reduction and broadcast transformations are handled separately.
//! It is recommend to use the composite ops view function, which will call
//! the analyzeView function to generate the appropriate transformations.
//!
//! For example:
//! original sizes = [2, 10, 40]
//! new_size = [2, 10, 2, 20]
//! auto analysis = analyzeView(TV0, original_sizes, new_sizes)
//! auto TV1 = TV0->view(analysis.transforms);
//!
//! Transforms = [(Keep I0), (Keep I1), (Split I2 by 2)]
//! Before: TV0[I0, I1, I2]
//! After: TV0[I0, I1, 2, ceilDiv(I2, 2)]
//!
TensorView* applyViewTransforms(
    TensorView* tv,
    const std::vector<std::shared_ptr<ViewTransform>>& transforms) {
  TORCH_INTERNAL_ASSERT(
      !tv->hasComputeAt(),
      "Cannot modify rfactor domain after compute at has been set.");

  TORCH_INTERNAL_ASSERT(tv->nDims() > 0, "Tried to view a 0-dim TensorView");

  TORCH_CHECK(
      !tv->domain()->hasRFactor(),
      "Cannot call view on the same TensorView twice.");

  TORCH_INTERNAL_ASSERT(!transforms.empty());

  TensorView* consumer =
      new TensorView(tv->domain()->view(transforms), tv->getDataType().value());

  new ViewOp(consumer, tv);

  return consumer;
}

} // namespace

TensorView* view(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  auto analyze_view = analyzeView(x, original_sizes, new_sizes);

  auto reduction = (!analyze_view.trivial_reduction_axes.empty())
      ? sum(x, analyze_view.trivial_reduction_axes)
      : x;

  auto view = (!analyze_view.transforms.empty())
      ? applyViewTransforms(reduction, analyze_view.transforms)
      : reduction;

  return (analyze_view.has_broadcast)
      ? broadcast(view, analyze_view.broadcast_axes)
      : view;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
