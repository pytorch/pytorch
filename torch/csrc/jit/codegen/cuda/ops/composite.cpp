#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/composite.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

ForwardDropoutResult dropout(TensorView* x, Val* prob) {
  auto p1m = sub(IrBuilder::create<Double>(x->container(), 1.), prob);
  auto zero_check =
      add(eq(p1m, IrBuilder::create<Double>(x->container(), 0.)), p1m);
  auto scale = div(IrBuilder::create<Double>(x->container(), 1.), zero_check);
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

  auto rand_vals = rand_like(x);
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

namespace {
template <typename T>
TORCH_CUDA_CU_API T* sign(T* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  auto zero = IrBuilder::create<Double>(x->container(), 0.);
  auto one = IrBuilder::create<Double>(x->container(), 1.);
  auto minus_one = IrBuilder::create<Double>(x->container(), -1.);
  auto sign = where(gt(x, zero), one, where(lt(x, zero), minus_one, zero));
  return castOp(x->getDataType().value(), sign);
}
} // namespace

TORCH_CUDA_CU_API TensorView* sign(TensorView* x) {
  return sign<TensorView>(x);
}

TORCH_CUDA_CU_API Val* sign(Val* x) {
  return sign<Val>(x);
}

TensorView* softplus(TensorView* x, Val* beta, Val* threshold) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(beta != nullptr, "Beta is invalid.");
  TORCH_INTERNAL_ASSERT(
      threshold != nullptr, "Threshold is not a valid Double.");

  auto op_beta = mul(x, beta);
  auto maybe_result = div(log1p(exp(op_beta)), beta);
  auto y = where(gt(op_beta, threshold), x, maybe_result);
  return y;
}

TensorView* gelu(TensorView* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  auto kappa = IrBuilder::create<Double>(x->container(), M_SQRT1_2);
  auto half = IrBuilder::create<Double>(x->container(), 0.5);
  auto one = IrBuilder::create<Double>(x->container(), 1.);

  auto cdf = mul(half, add(one, erf(mul(x, kappa))));
  auto y = mul(x, cdf);
  return y;
}

TensorView* gelu_backward(TensorView* dy, TensorView* x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  const double kHalf = 0.5;

  auto cdf_1 = mul(x, IrBuilder::create<Double>(x->container(), M_SQRT1_2));
  auto cdf_2 = erf(cdf_1);
  auto cdf_3 = add(cdf_2, IrBuilder::create<Double>(x->container(), 1.));
  auto cdf_4 = mul(cdf_3, IrBuilder::create<Double>(x->container(), kHalf));

  auto pdf_1 = mul(x, x);
  auto pdf_2 = mul(pdf_1, IrBuilder::create<Double>(x->container(), -kHalf));
  auto pdf_3 = exp(pdf_2);

  auto out = addcmul(
      cdf_4, x, pdf_3, IrBuilder::create<Double>(x->container(), kAlpha));
  auto dx = mul(out, dy);
  return dx;
}

TensorView* tanh_gelu(TensorView* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_cube = mul(x, mul(x, x));

  auto inner_1 = mul(IrBuilder::create<Double>(x->container(), kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(IrBuilder::create<Double>(x->container(), kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto out =
      mul(x, add(IrBuilder::create<Double>(x->container(), 1.), tanh_inner));
  auto y = mul(IrBuilder::create<Double>(x->container(), 0.5), out);
  return y;
}

TensorView* tanh_gelu_backward(TensorView* dy, TensorView* x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid");

  constexpr double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr double kKappa = 0.044715;

  auto x_sq = mul(x, x);
  auto x_cube = mul(x, x_sq);

  auto inner_1 = mul(IrBuilder::create<Double>(x->container(), kKappa), x_cube);
  auto inner_2 = add(x, inner_1);
  auto inner_3 = mul(IrBuilder::create<Double>(x->container(), kBeta), inner_2);
  auto tanh_inner = tanh(inner_3);

  auto left = mul(IrBuilder::create<Double>(x->container(), 0.5), x);
  auto right = add(IrBuilder::create<Double>(x->container(), 1.), tanh_inner);

  auto left_derivative =
      mul(IrBuilder::create<Double>(x->container(), 0.5), right);

  auto tanh_inner_sq = mul(tanh_inner, tanh_inner);
  auto tanh_derivative =
      sub(IrBuilder::create<Double>(x->container(), 1), tanh_inner_sq);

  auto constant_mul_x_sq =
      mul(IrBuilder::create<Double>(x->container(), kBeta * 3 * kKappa), x_sq);
  auto inner_derivative =
      add(IrBuilder::create<Double>(x->container(), kBeta), constant_mul_x_sq);
  auto right_derivative = mul(left, mul(tanh_derivative, inner_derivative));

  auto dx = mul(dy, add(left_derivative, right_derivative));
  return dx;
}

TensorView* tanh_backward(TensorView* dy, TensorView* tanh_x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(tanh_x != nullptr, "Input is invalid");

  auto one = IrBuilder::create<Double>(tanh_x->container(), 1.);
  auto tanh_sq = mul(tanh_x, tanh_x);
  auto sub_tanh_sq = sub(one, tanh_sq);
  auto dx = mul(dy, sub_tanh_sq);
  return dx;
}

TensorView* leaky_relu(TensorView* x, Val* negative_slope) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "input is invalid.");
  TORCH_INTERNAL_ASSERT(negative_slope != nullptr, "negative_slope is invalid");
  auto zero = IrBuilder::create<Double>(x->container(), 0.);
  return where(ge(x, zero), x, mul(negative_slope, x));
}

TensorView* view_as_real(TensorView* x) {
  auto input_type = x->getDataType().value();
  TORCH_CHECK(
      isComplexType(input_type),
      "Operand of view_as_real must have complex type");

  auto vec_type = getVectorType(getTypeFromComplexType(input_type), 2);
  auto tv_vector = bitCastOp(vec_type, x);
  return viewAsScalar(tv_vector);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
