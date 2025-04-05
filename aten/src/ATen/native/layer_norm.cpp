#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/layer_norm.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/layer_norm_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/native_rms_norm_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#ifdef USE_MPS
#include <ATen/native/mps/operations/RMSNorm.h>
#include <c10/core/GradMode.h>
#endif

#include <array>
#include <tuple>
#include <vector>

namespace at::native {

static void layer_norm_with_mean_rstd_out(
    at::Tensor& out,
    at::Tensor& mean,
    at::Tensor& rstd,
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, &mean, &rstd);
  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  DimVector stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.emplace_back(input_shape[idx]);
  }
  for ([[maybe_unused]] const auto idx : c10::irange(axis, input.dim())) {
    stat_shape.emplace_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
}

void layer_norm_cpu_out(
    at::Tensor& out,
    const at::Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    double eps,
    int64_t M,
    int64_t N) {
  if (M <= 0) {
    return;
  }
  LayerNormKernel(kCPU, input, gamma, beta, M, N, eps, &out, /*mean=*/nullptr, /*rstd=*/nullptr);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& input,
    IntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  bool mixed_type = is_mixed_type(input, weight, bias);
  if (mixed_type) {
    check_mixed_data_type(input, weight, bias);
  }

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  const auto dtype = param_scalar_type(input, mixed_type);
  Tensor mean = at::empty({M}, X->options().dtype(dtype));
  Tensor rstd = at::empty({M}, X->options().dtype(dtype));

  layer_norm_with_mean_rstd_out(Y, mean, rstd, *X, normalized_shape, *gamma, *beta, eps, M, N);
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        *X,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  if (M > 0) {
    LayerNormBackwardKernel(
        kCPU, dY, *X, mean, rstd, *gamma, M, N, &dX, &dgamma, &dbeta);
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

Tensor layer_norm_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  return std::get<0>(at::native_layer_norm_symint(input, normalized_shape, weight_opt, bias_opt, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

// Ported from pytorch/xla repo
std::tuple<Tensor, Tensor, Tensor> math_native_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();

  auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int axis = input_ndim - normalized_ndim;

  // Properly handle zero-size inputs: the view(1, M, -1) call below breaks on this.
  if (input.numel() == 0) {
    auto result_type = c10::promoteTypes(input.scalar_type(), kFloat);
    return std::make_tuple(
      at::empty_like(input),
      at::empty_like(input, c10::TensorOptions().dtype(result_type)),
      at::empty_like(input, c10::TensorOptions().dtype(result_type))
    );
  }
  at::Tensor input_reshaped = input.reshape({1, M, -1});
  // Unlike Batch Normalization, which applies scalar scale and bias for each
  // entire channel/plane with the affine option, Layer Normalization applies
  // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
  // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  auto& [out, mean, rstd] = outputs;
  out = out.view(input_shape);
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    out = out.mul(weight);
  } else if (bias.defined()) {
    out = out.add(bias);
  }
  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for ([[maybe_unused]] const auto idx : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  return outputs;
}

Tensor rms_norm_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    std::optional<double> eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  _check_rms_norm_inputs_symint(input, normalized_shape, weight);

#ifdef USE_MPS
  if (input.device().type() == DeviceType::MPS && weight_opt.has_value()) {
    const Tensor weight = weight_opt.value();
    const bool any_nested = input.is_nested() || weight.is_nested();
    const bool any_inputs_require_grad = input.requires_grad() || weight.requires_grad();
    const bool is_input_fp = isFloatingType(input.scalar_type());
    const bool is_weight_fp = isFloatingType(weight.scalar_type());

    if (!(GradMode::is_enabled() && any_inputs_require_grad) && !any_nested && is_input_fp && is_weight_fp) {
      auto eps_val = eps.value_or(std::numeric_limits<double>::epsilon());
      return mps::rms_norm_mps_kernel(input.contiguous(), normalized_shape, weight.contiguous(), eps_val);
    }
  }
#endif

  std::vector<int64_t> dims_to_reduce;
  for (const auto i : c10::irange(normalized_shape.size())) {
    dims_to_reduce.push_back(input.dim() - i - 1);
  }
  IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  auto result = AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "rms_norm",
        [&] {
    // upcast is needed for fp16 and bf16
    c10::ScalarType opmath_t = toOpMathType(input.scalar_type());
    Tensor upcasted_input = input.to(opmath_t);

    Tensor rqrst_input;

    // opmath_t would be one of [Double, Float, ComplexFloat, ComplexDouble]
    if (opmath_t == at::ScalarType::Float || opmath_t == at::ScalarType::ComplexFloat) {
      using limits = std::numeric_limits<float>;
      float eps_val = eps.value_or(limits::epsilon());
      rqrst_input = rsqrt(at::pow(upcasted_input, 2).mean(dims_to_reduce_ref, /*keepdim=*/true).add_(eps_val));
    } else {
      using limits = std::numeric_limits<double>;
      double eps_val = eps.value_or(limits::epsilon());
      rqrst_input = rsqrt(at::pow(upcasted_input, 2).mean(dims_to_reduce_ref, /*keepdim=*/true).add_(eps_val));
    }

    Tensor upcasted_result = upcasted_input.mul(rqrst_input);

    if (weight_opt.has_value()) {
      upcasted_result = upcasted_result.mul(weight_opt.value());
    }

    return upcasted_result;
  });

  return result.type_as(input);

}

std::tuple<Tensor, Tensor, Tensor> rms_norm_cpu(
  at::Tensor const& input,
  c10::ArrayRef<long> normalized_shape,
  std::optional<at::Tensor> const& weight_opt,
  std::optional<double> eps_opt) {
  // Use a default epsilon if not provided
  double eps = eps_opt.has_value() ? eps_opt.value() : 1e-6;

  // Calculate the number of elements in the normalized dimensions.
  int64_t norm_numel = 1;
  for (auto dim : normalized_shape) {
    norm_numel *= dim;
  }

  // Square the input.
  auto squared = input.pow(2);

  // Determine the dimensions to reduce over.
  // Assuming normalized_shape corresponds to the trailing dimensions.
  int64_t input_dims = input.dim();
  int64_t num_normalized_dims = normalized_shape.size();
  std::vector<int64_t> reduce_dims;
  for (int64_t i = input_dims - num_normalized_dims; i < input_dims; i++) {
    reduce_dims.push_back(i);
  }

  // Compute the mean over the normalized dimensions and keep dimensions for broadcasting.
  auto mean_squared = squared.mean(reduce_dims, /*keepdim=*/true);

  // Compute the inverse RMS: 1 / sqrt(mean_squared + eps)
  auto inv_rms = at::rsqrt(mean_squared.add_(eps));

  // Normalize the input.
  auto x_norm = at::mul(input, inv_rms);

  // If a weight is provided, apply it elementwise; otherwise, output is just x_norm.
  at::Tensor output;
  if (weight_opt.has_value() && weight_opt.value().defined()) {
    // Assumes weight is broadcastable to the shape of x_norm.
    output = at::mul(x_norm, weight_opt.value());
  } else {
    output = x_norm;
  }
  // Return the tuple: (output, x_norm, inv_rms)
  return std::make_tuple(output, x_norm, inv_rms);
}

std::tuple<Tensor, Tensor>
rms_norm_backward_cpu(
  const Tensor &grad,                      // Gradient with respect to the output y
  const Tensor &input,                     // Original input x
  const std::optional<Tensor> &weight,       // Optional weight (scaling factor)
  const std::optional<double> eps,         // Optional epsilon (not used here as inverse_rms is provided)
  const Tensor &output,                    // Output from the forward pass (unused in this backward computation)
  const Tensor &x_norm,                    // Normalized input (x multiplied by inverse_rms)
  const Tensor &inverse_rms,               // Inverse RMS factor: 1 / sqrt(mean(x^2) + eps)
  std::array<bool, 2ul> grad_input_mask    // Mask: [compute_grad_input, compute_grad_weight]
) {
  // Initialize outputs as empty tensors.
  Tensor grad_input = at::Tensor();
  Tensor grad_weight = at::Tensor();
  
  // Assume normalization over the last dimension.
  const int64_t D = input.size(-1);
  const double D_float = static_cast<double>(D);
  
  // Compute gradient with respect to input if requested.
  if (grad_input_mask[0]) {
    // If weight is provided, use it to scale grad; otherwise, use grad directly.
    Tensor grad_weighted = weight.has_value() ? at::mul(grad, weight.value()) : grad;
    
    // Compute dot = sum(grad_weighted * input) over the last dimension, keeping the dimension.
    Tensor dot = at::sum(at::mul(grad_weighted, input), -1, /*keepdim=*/true);
    
    // Compute r^3 using at::pow.
    Tensor r_cubed = at::pow(inverse_rms, 3);
    
    // Compute the first term: grad_weighted * inverse_rms.
    Tensor term1 = at::mul(grad_weighted, inverse_rms);
    
    // Compute the second term: input * (r^3 * (dot / D)).
    Tensor dot_div = at::div(dot, D_float);
    Tensor term2 = at::mul(input, at::mul(r_cubed, dot_div));
    
    // Combine the terms to get grad_input.
    grad_input = at::sub(term1, term2);
  }
  
  // Compute gradient with respect to weight if requested and if weight is provided.
  if (grad_input_mask[1] && weight.has_value()) {
    // For weight gradient, sum over all dimensions except the normalized one.
    std::vector<int64_t> reduce_dims;
    for (int64_t i = 0; i < input.dim() - 1; ++i) {
      reduce_dims.push_back(i);
    }
    grad_weight = at::sum(at::mul(grad, x_norm), reduce_dims);
  }
  
  return std::make_tuple(grad_input, grad_weight);
}

DEFINE_DISPATCH(RMSNormKernel);


} // namespace at::native
