#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/group_norm.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/full_native.h>
#include <ATen/ops/group_norm_native.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward_native.h>
#include <ATen/ops/native_group_norm_native.h>
#include <ATen/ops/var_mean.h>
#endif

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace at::native {

template <typename T>
static void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const T& C,
    int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ", num_groups);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& X,
    const std::optional<Tensor>& gamma_opt /* optional */,
    const std::optional<Tensor>& beta_opt /* optional */,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  const Tensor& beta = beta_opt.value_or(Tensor());

  // repeated check so expanded weights can call native_group_norm directly but
  // save mean and variance from forward
  check_group_norm_inputs(X, gamma, beta, C, group);
  auto memory_format = X.device().is_cpu() ?
      X.suggest_memory_format() : at::MemoryFormat::Contiguous;

  TORCH_CHECK(X.is_contiguous(memory_format));

  bool mixed_type = is_mixed_type(X, gamma, beta);
  if (mixed_type) {
    check_mixed_data_type(X, gamma, beta);
  }

  Tensor Y = at::native::empty_like(
      X,
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      memory_format);
  const auto dtype = param_scalar_type(X, mixed_type);
  Tensor mean = at::empty({N, group}, X.options().dtype(dtype));
  Tensor rstd = at::empty({N, group}, X.options().dtype(dtype));
  GroupNormKernel(
      X.device().type(), X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  TORCH_CHECK(
      X.scalar_type() == dY.scalar_type(),
      "Expected scalar types of X and dY are same.");
  bool mixed_type = is_mixed_type(X, mean, rstd);
  if (mixed_type) {
    check_mixed_data_type(X, mean, rstd);
  }
  auto memory_format = X.device().is_cpu() ?
      X.suggest_memory_format() : at::MemoryFormat::Contiguous;

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        memory_format);
  }
  if (grad_input_mask[1]) {
    dgamma = at::native::empty_like(
        gamma,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = at::native::empty_like(
        gamma,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  GroupNormBackwardKernel(
      X.device().type(),
      dY,
      X,
      mean,
      rstd,
      gamma,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta);
  return std::make_tuple(dX, dgamma, dbeta);
}

Tensor group_norm(
    const Tensor& input,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enabled, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());

  const auto N = input.sym_size(0);
  const auto C = input.sym_size(1);
  check_group_norm_inputs(input, weight, bias, C, num_groups);

  const auto input_shape = input.sym_sizes();
  const auto HxW =
      c10::multiply_integers(input_shape.slice(2));

  const Tensor kEmpty;
  auto memory_format = input.suggest_memory_format();
  const auto& X = input.device().is_cpu() || input.is_privateuseone() ?
                  input.contiguous(memory_format) : input.contiguous();
  const auto& gamma = weight.defined() ? weight.contiguous() : kEmpty;
  const auto& beta = bias.defined() ? bias.contiguous() : kEmpty;
  TORCH_CHECK(!gamma.defined() || gamma.sym_numel() == C);
  TORCH_CHECK(!beta.defined() || beta.sym_numel() == C);
  return std::get<0>(
      at::native_group_norm_symint(X, gamma, beta, N, C, HxW, num_groups, eps));
}

DEFINE_DISPATCH(GroupNormKernel);
DEFINE_DISPATCH(GroupNormBackwardKernel);

std::tuple<at::Tensor, at::Tensor, at::Tensor> math_group_norm(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  auto input_shape = input.sizes();
  if (std::ranges::any_of(input_shape, [](auto s) { return s == 0; })) {
    return std::make_tuple(
        at::native::empty_like(input),
        at::native::full({N, group}, NAN, input.scalar_type(), {}, input.device()),
        at::native::full({N, group}, NAN, input.scalar_type(), {}, input.device()));
  }

  auto input_reshaped = input.view({N, group, C / group * HxW});
  auto [var, mean] = at::var_mean(input_reshaped, {2}, c10::Scalar(0), true);
  auto rsqrt = var.add(eps).rsqrt();
  auto out = input_reshaped.sub(mean).mul(rsqrt).reshape(input_shape);

  std::vector<int64_t> weight_bias_shape(input.ndimension(), 1);
  weight_bias_shape[1] = C;
  if (weight_opt && weight_opt->defined()) {
    out = out.mul(weight_opt->view(weight_bias_shape));
  }
  if (bias_opt && bias_opt->defined()) {
    out = out.add(bias_opt->view(weight_bias_shape));
  }

  mean = mean.squeeze(-1);
  rsqrt = rsqrt.squeeze(-1);
  return std::make_tuple(std::move(out), std::move(mean), std::move(rsqrt));
}
} // namespace at::native
