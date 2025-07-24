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
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/layer_norm_native.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#ifdef USE_MPS
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

std::tuple<Tensor, Tensor> rms_norm_composite(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    std::optional<double> eps) {

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

    // if nested do not make contiguous
    if(input.is_nested() || (weight_opt.has_value() && weight_opt.value().is_nested())){
      return std::make_tuple(upcasted_result, rqrst_input);
    }

    if(input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast || input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d){
      return std::make_tuple(upcasted_result, rqrst_input);
    }

    return std::make_tuple(upcasted_result.contiguous(), rqrst_input.contiguous());
  });
  return std::make_tuple(
    std::get<0>(result).type_as(input), // Cast normalized result to original input type
    std::get<1>(result)                 // rsqrt_val
  );
}


Tensor rms_norm_symint(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<double> eps) {

  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  _check_rms_norm_inputs_symint(input, normalized_shape, weight);

  // composite fallback for channels last
  if(input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast || input.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d){
    return std::get<0>(rms_norm_composite(input, IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
  }

  // composite fallback for complex datatypes
  if(input.is_complex()){
    return std::get<0>(rms_norm_composite(input, IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
  }

  if (weight_opt.has_value() && weight_opt.value().defined() && weight_opt.value().dtype() != input.dtype()) {
    TORCH_WARN_ONCE(
      "Mismatch dtype between input and module: input dtype = ", input.dtype(),
      ", module dtype = ", weight_opt.value().dtype(), ", Can not dispatch to fused implementation"
    );
    return std::get<0>(rms_norm_composite(input, IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
  }

  #ifdef USE_MPS
  if (input.device().type() == DeviceType::MPS && weight_opt.has_value()) {
    const Tensor weight = weight_opt.value();
    const bool any_inputs_require_grad = input.requires_grad() || weight.requires_grad();

    if (!(GradMode::is_enabled() && any_inputs_require_grad)) {
      return std::get<0>(at::_fused_rms_norm(input.contiguous(), IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
    }
  }

  if (input.device().type() == DeviceType::MPS){
    return std::get<0>(rms_norm_composite(input, IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
  }
  #endif
  return std::get<0>(at::_fused_rms_norm(input, IntArrayRef(reinterpret_cast<const int64_t*>(normalized_shape.data()), normalized_shape.size()), weight_opt, eps));
}

} // namespace at::native
