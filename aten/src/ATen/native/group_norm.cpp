#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/group_norm.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/full.h>
#include <ATen/ops/group_norm_native.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/native_group_norm_backward_native.h>
#include <ATen/ops/native_group_norm_native.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <array>
#include <tuple>
#include <vector>

namespace at::native {

static MemoryFormat group_norm_memory_format(const Tensor& input) {
  return (input.device().is_cpu() || input.device().is_cuda() ||
          input.device().is_privateuseone())
      ? input.suggest_memory_format()
      : MemoryFormat::Contiguous;
}

template <typename T>
static void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const T& C,
    const T& HxW,
    int64_t num_groups) {
  // We explicitly support N == 0, but if either C or HxW == 0 the results are
  // non-sensical.
  TORCH_CHECK(
      C > 0, "Expected number of channels to be greater than 0, got ", C);
  TORCH_CHECK(HxW > 0, "Expected HxW to be greater than 0, got ", HxW);
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ",
      num_groups);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() ||
          (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      bias.sizes(),
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
  c10::MaybeOwned<Tensor> beta_maybe_owned =
      at::borrow_from_optional_tensor(beta_opt);
  const Tensor& gamma = *gamma_maybe_owned;
  const Tensor& beta = *beta_maybe_owned;

  // repeated check so expanded weights can call native_group_norm directly but
  // save mean and variance from forward
  check_group_norm_inputs(X, gamma, beta, C, HxW, group);
  bool mixed_type = is_mixed_type(X, gamma, beta);
  if (mixed_type) {
    check_mixed_data_type(X, gamma, beta);
  }

  auto memory_format{group_norm_memory_format(X)};
  auto stat_options{X.options()
                        .memory_format(at::MemoryFormat::Contiguous)
                        .dtype(param_scalar_type(X, mixed_type))};

  Tensor Y = at::empty_like(X, {}, memory_format);
  Tensor mean = at::empty({N, group}, stat_options);
  Tensor rstd = at::empty({N, group}, stat_options);

  if (N) {
    Tensor X_ = X.contiguous(memory_format);
    Tensor gamma_ = gamma.defined() ? gamma.contiguous() : gamma;
    Tensor beta_ = beta.defined() ? beta.contiguous() : beta;

    GroupNormKernel(
        X_.device().type(),
        X_,
        gamma_,
        beta_,
        N,
        C,
        HxW,
        group,
        eps,
        Y,
        mean,
        rstd);
  }

  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
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

  auto memory_format = group_norm_memory_format(X);
  auto dparam_options{(gamma.defined() ? gamma.options() : X.options())
                          .memory_format(MemoryFormat::Contiguous)};

  if (!N) {
    return std::make_tuple(
        grad_input_mask[0] ? at::zeros_like(X, {}, memory_format) : Tensor{},
        grad_input_mask[1] ? at::zeros({C}, dparam_options) : Tensor{},
        grad_input_mask[2] ? at::zeros({C}, dparam_options) : Tensor{});
  }

  auto dY_{dY.contiguous(memory_format)};
  auto X_{X.contiguous(memory_format)};
  auto mean_{mean.contiguous()};
  auto rstd_{rstd.contiguous()};
  auto gamma_{gamma.defined() ? gamma.contiguous() : gamma};

  Tensor dX{};
  if (grad_input_mask[0]) {
    dX = at::empty_like(X_);
  }
  Tensor dgamma{};
  if (grad_input_mask[1]) {
    dgamma = at::empty({C}, dparam_options);
  }
  Tensor dbeta{};
  if (grad_input_mask[2]) {
    dbeta = at::empty({C}, dparam_options);
  }
  GroupNormBackwardKernel(
      X_.device().type(),
      dY_,
      X_,
      mean_,
      rstd_,
      gamma_,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta);
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
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
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = *bias_maybe_owned;

  const auto N = input.sym_size(0);
  const auto C = input.sym_size(1);
  const auto input_shape = input.sym_sizes();
  const auto HxW = c10::multiply_integers(input_shape.slice(2));
  check_group_norm_inputs(input, weight, bias, C, HxW, num_groups);

  return std::get<0>(at::native_group_norm_symint(
      input, weight, bias, N, C, HxW, num_groups, eps));
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
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = *bias_maybe_owned;

  check_group_norm_inputs(input, weight, bias, C, HxW, group);

  auto memory_format{input.suggest_memory_format()};
  auto stat_options{
      input.options().memory_format(at::MemoryFormat::Contiguous)};

  if (!N) {
    return std::make_tuple(
        at::empty_like(input, {}, memory_format),
        // Return in the dtype of input, matching the operations below, unlike
        // the optimized native_group_norm impl above.
        at::empty({N, group}, stat_options),
        at::empty({N, group}, stat_options));
  }

  auto input_reshaped = input.reshape({N, group, C / group * HxW});
  auto [var, mean] = at::var_mean(input_reshaped, {2}, c10::Scalar(0), true);
  auto rsqrt = var.add(eps).rsqrt();
  auto out = input_reshaped.sub(mean).mul(rsqrt).reshape(input.sizes());

  std::vector<int64_t> weight_bias_shape(input.ndimension(), 1);
  weight_bias_shape[1] = C;
  if (weight_opt && weight_opt->defined()) {
    out = out.mul(weight_opt->view(weight_bias_shape));
  }
  if (bias_opt && bias_opt->defined()) {
    out = out.add(bias_opt->view(weight_bias_shape));
  }

  out = out.contiguous(memory_format);
  mean = mean.squeeze(-1);
  rsqrt = rsqrt.squeeze(-1);
  return std::make_tuple(std::move(out), std::move(mean), std::move(rsqrt));
}
} // namespace at::native
