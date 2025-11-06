// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at::functorch {

static bool is_empty_tensor(const Tensor& tensor) {
  const auto shape = tensor.sizes();
  return shape.size() == 1 && shape[0] == 0;
}

static std::optional<int64_t> compute_stat_bdim(
    std::optional<int64_t> input_bdim,
    const Tensor& stat) {
  // There's a weird case where mean, rstd can both have shape (0,).
  // It's possible that this is a bug on the PyTorch side.
  // When that happens we don't want to return a BatchedTensor.
  if (input_bdim.has_value() && !is_empty_tensor(stat)) {
    return 0;
  }
  return std::nullopt;
}

static Tensor padRight(const Tensor& tensor, std::optional<int64_t> has_bdim, int64_t logical_rank) {
  // NB: Batch dim, if it exists, is assumed to be the first dim
  auto tensor_logical_rank = rankWithoutBatchDim(tensor, has_bdim);
  if (tensor_logical_rank >= logical_rank) {
    return tensor;
  }
  VmapDimVector new_sizes(tensor.sizes().begin(), tensor.sizes().end());
  for (int64_t i = 0; i < logical_rank - tensor_logical_rank; i++) {
    new_sizes.push_back(1);
  }
  return tensor.view(new_sizes);
}

template<typename F, F Func>
static
std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
batch_norm_batch_rule(
    const Tensor& input, std::optional<int64_t> input_bdim,
    const std::optional<Tensor>& weight_opt, std::optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, std::optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, std::optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, std::optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const auto& running_mean = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
  const auto& running_var = *running_var_maybe_owned;
  TORCH_CHECK(!training || (!input_bdim || ((!running_mean.defined() || running_mean_bdim) && (!running_var.defined() || running_var_bdim))),
      "Batch norm got a batched tensor as input while the running_mean or running_var, which will be updated in place, ",
      "were not batched.\nIf you are using a module and do not need eval mode, please set `track_running_stats` to be False.",
      "If you are using a prebuilt module and do not need eval mode, please see the functorch website for resources on ",
      "how to patch your module to work with vmap");
  std::optional<int64_t> bdim_size;
  Tensor result0;
  Tensor mean;
  Tensor rstd;
  if (!input_bdim && !running_mean_bdim && !running_var_bdim) {
    const auto dummy_weight = at::ones(input.size(1), input.options());  // cudnn and miopen require a weight
    const auto dummy_bias = at::zeros(input.size(1), input.options());   // without this, get "strides() called on undefined Tensor" on cuda
    auto result = Func(input, dummy_weight, dummy_bias, running_mean_opt, running_var_opt, training, momentum, eps);
    result0 = std::get<0>(result).transpose(0, 1);          // [C, B, *]
    mean = std::move(std::get<1>(result));
    rstd = std::move(std::get<2>(result));
  } else {
    bdim_size = get_bdim_size3(input, input_bdim, running_mean, running_mean_bdim, running_var, running_var_bdim);
    auto input_ = moveBatchDimToFront(input, input_bdim);
    input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size.value());
    input_ = reshape_dim_into(0, /*channels dim*/1, input_);

    std::optional<Tensor> running_mean_;
    std::optional<Tensor> running_var_;
    if (running_mean.defined()) {
      running_mean_ = moveBatchDimToFront(running_mean, running_mean_bdim);
      running_mean_ = ensure_has_bdim(*running_mean_, running_mean_bdim.has_value(), bdim_size.value());
      running_mean_ = reshape_dim_into(0, 0, *running_mean_).contiguous();
    }
    if (running_var.defined()) {
      running_var_ = moveBatchDimToFront(running_var, running_var_bdim);
      running_var_ = ensure_has_bdim(*running_var_, running_var_bdim.has_value(), bdim_size.value());
      running_var_ = reshape_dim_into(0, 0, *running_var_).contiguous();
    }

    const auto dummy_weight = at::ones(input_.size(1), input_.options());  // cudnn and miopen require a weight
    const auto dummy_bias = at::zeros(input_.size(1), input_.options());   // without this, get "strides() called on undefined Tensor" on cuda
    auto result = Func(input_, dummy_weight, dummy_bias, running_mean_, running_var_, training, momentum, eps);
    result0 = std::get<0>(result).transpose(0, 1);                // [(B0, C), B, *]
    mean = std::move(std::get<1>(result));
    rstd = std::move(std::get<2>(result));
    result0 = reshape_dim_outof(0, bdim_size.value(), result0);   // [B0, C, B, *]
    mean = reshape_dim_outof(0, bdim_size.value(), mean);         // [B0, C]
    rstd = reshape_dim_outof(0, bdim_size.value(), rstd);         // [B0, C]
  }

  const auto stats_bdim = compute_stat_bdim(bdim_size, mean);
  if (weight.defined()) {
    const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
    auto weight_ = moveBatchDimToFront(weight, weight_bdim);
    weight_ = padRight(weight_, weight_bdim, input_logical_rank);
    result0 = result0 * weight_;
  }
  if (bias.defined()) {
    const auto result_logical_rank = rankWithoutBatchDim(
        result0,
        bdim_size.has_value() || weight_bdim.has_value() ? std::optional<int64_t>(0) : std::optional<int64_t>(std::nullopt));
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    bias_ = padRight(bias_, bias_bdim, result_logical_rank);
    result0 = result0 + bias_;
  }
  result0 = result0.transpose(1, 2);  // [B0, B, C, *], because some arg must have been batched, the output must be batched
  return std::make_tuple(std::move(result0), 0, std::move(mean), stats_bdim, std::move(rstd), stats_bdim);
}

template<typename F, F Func>
static
std::tuple<at::Tensor, std::optional<int64_t>> batch_norm_backward_no_weight_bias_batch_rule(
    const at::Tensor & grad_out, std::optional<int64_t> grad_out_bdim,
    const at::Tensor & input, std::optional<int64_t> input_bdim,
    const std::optional<at::Tensor> & running_mean_opt, std::optional<int64_t> running_mean_bdim,
    const std::optional<at::Tensor> & running_var_opt, std::optional<int64_t> running_var_bdim,
    const at::Tensor & mean, std::optional<int64_t> mean_bdim,
    const at::Tensor & rstd, std::optional<int64_t> rstd_bdim,
    bool training, double eps) {
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
  const Tensor& running_var = *running_var_maybe_owned;

  if (!grad_out_bdim.has_value() && !input_bdim.has_value() && !running_mean_bdim.has_value() && !running_var_bdim.has_value()) {
    // for either of these to have bdims, the input, running_mean, or running_var must have had a bdim
    TORCH_INTERNAL_ASSERT(!mean_bdim);
    TORCH_INTERNAL_ASSERT(!rstd_bdim);
    const auto dummy_weight = at::ones(input.size(1), input.options());
    auto result =Func(
        grad_out, input, dummy_weight, running_mean_opt, running_var_opt, mean, rstd, training, eps, {true, false, false});
    return {std::move(std::get<0>(result)), std::nullopt};
  }

  auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto mean_ = moveBatchDimToFront(mean, mean_bdim);
  auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

  // ensure all inputs have bdim.
  const auto bdim_size = get_bdim_size4(grad_out, grad_out_bdim, input, input_bdim, running_mean, running_mean_bdim, running_var, running_var_bdim);
  grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), bdim_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
  mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
  rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

  std::optional<Tensor> running_mean_;
  std::optional<Tensor> running_var_;
  if (running_mean.defined()) {
    running_mean_ = moveBatchDimToFront(running_mean, running_mean_bdim);
    running_mean_ = ensure_has_bdim(*running_mean_, running_mean_bdim.has_value(), bdim_size);
    running_mean_ = reshape_dim_into(0, 0, *running_mean_).contiguous();
  }
  if (running_var.defined()) {
    running_var_ = moveBatchDimToFront(running_var, running_var_bdim);
    running_var_ = ensure_has_bdim(*running_var_, running_var_bdim.has_value(), bdim_size);
    running_var_ = reshape_dim_into(0, 0, *running_var_).contiguous();
  }

  input_ = reshape_dim_into(0, /*channels dim*/1, input_);
  TORCH_INTERNAL_ASSERT(mean_.dim() == 2);
  TORCH_INTERNAL_ASSERT(rstd_.dim() == 2);
  mean_ = reshape_dim_into(0, 0, mean_);
  rstd_ = reshape_dim_into(0, 0, rstd_);
  grad_out_ = grad_out_.transpose(0, 1).flatten(1, 2); // [B0, B, C, *] -> [B, (B0, C), *]

  const auto dummy_weight = at::ones(input_.size(1), input_.options());
  auto result = at::native_batch_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      dummy_weight,
      running_mean_,  // contiguous called if there is a tensor given
      running_var_,   // contiguous called if there is a tensor given
      mean_.contiguous(),
      rstd_.contiguous(),
      training, eps, {true, false, false});
  auto& result0 = std::get<0>(result);
  result0 = reshape_dim_outof(1, bdim_size, result0); // [B, B0, C, *]
  result0 = result0.transpose(0, 1); // [B0, B, C, *]
  return std::make_tuple(std::move(result0), 0);
}

template<typename F, F Func>
static
std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_plumbing(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const std::optional<at::Tensor> & weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
  const Tensor& running_var = *running_var_maybe_owned;
  // NB: not sure why these are optional...these are required from the forward
  TORCH_INTERNAL_ASSERT(save_mean_opt.has_value());
  TORCH_INTERNAL_ASSERT(save_rstd_opt.has_value());
  const Tensor& save_mean = *save_mean_opt;
  const Tensor& save_rstd = *save_rstd_opt;
  TORCH_INTERNAL_ASSERT(save_mean.defined());
  TORCH_INTERNAL_ASSERT(save_rstd.defined());

  // plumbing
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "batch_norm_backward_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();

  auto [grad_out_value, grad_out_bdim] = unwrapTensorAtLevel(grad_out, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  Tensor mean_value;
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight.defined()) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }
  std::optional<Tensor> running_mean_value;
  std::optional<int64_t> running_mean_bdim;
  if (running_mean.defined()) {
    std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(running_mean, cur_level);
  }
  std::optional<Tensor> running_var_value;
  std::optional<int64_t> running_var_bdim;
  if (running_var.defined()) {
    std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(running_var, cur_level);
  }
  auto [save_mean_value, save_mean_bdim] = unwrapTensorAtLevel(save_mean, cur_level);
  auto [save_rstd_value, save_rstd_bdim] = unwrapTensorAtLevel(save_rstd, cur_level);

  // results
  Tensor grad_bias;
  Tensor grad_weight;
  Tensor grad_input;

  TORCH_INTERNAL_ASSERT(grad_out_value.dim() > 1);  // batch_norm can't operate on 1D tensors so the output will be at least 2D
  if (output_mask[2]) {
    grad_bias = grad_out.transpose(0, 1).sum(range(1, grad_out.dim()));
  }
  if (output_mask[1] && weight_value.has_value()) {
    // NB: output isn't saved...
    auto mean = training ? save_mean : running_mean;
    auto var = training ? save_rstd : (1 / at::sqrt(running_var + eps));
    const auto normalized_input = (input.transpose(0, 1) - padRight(mean, std::nullopt, input.dim())) * padRight(var, std::nullopt, input.dim());
    const auto expanded_grad_weight = normalized_input * grad_out.transpose(0, 1);
    grad_weight = expanded_grad_weight.sum(range(1, grad_out.dim()));
  }
  if (output_mask[0]) {
    const auto grad_normalized_input = weight.defined() ?
      grad_out.transpose(0, 1) * padRight(weight, std::nullopt, grad_out.dim()) : grad_out.transpose(0, 1);           // [B0, C, B, *]
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input.transpose(0, 1), cur_level);       // [B0, B, C, *]

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto results = batch_norm_backward_no_weight_bias_batch_rule<F, Func>(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        running_mean_value, running_mean_bdim,
        running_var_value, running_var_bdim,
        save_mean_value, save_mean_bdim,
        save_rstd_value, save_rstd_bdim,
        training, eps);
    grad_input = makeBatched(std::move(std::get<0>(results)), std::get<1>(results), cur_level);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<Tensor,Tensor,Tensor> native_group_norm_plumbing(
    const Tensor & input, const std::optional<Tensor> & weight_opt,
    const std::optional<Tensor> & bias_opt, int64_t N, int64_t C,
    int64_t HxW, int64_t group, double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "native_group_norm_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();

  if (!areAnyBatchedAtLevel({input, weight_opt, bias_opt}, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::native_group_norm(input, weight_opt, bias_opt, N, C, HxW, group, eps);
  }

  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);

  Tensor result0;
  Tensor mean;
  Tensor rstd;
  if (input_bdim) {
    const auto input_ = reshape_dim_into(*input_bdim, 0, input_value);
    const auto bdim_size = input_value.size(*input_bdim);

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    std::tie(result0, mean, rstd) = at::native_group_norm(input_, std::nullopt, std::nullopt, N * bdim_size, C, HxW, group, eps);
    result0 = makeBatched(reshape_dim_outof(0, bdim_size, result0), 0, cur_level);
    mean = makeBatched(reshape_dim_outof(0, bdim_size, mean), 0, cur_level);
    rstd = makeBatched(reshape_dim_outof(0, bdim_size, rstd), 0, cur_level);
  } else {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    std::tie(result0, mean, rstd) = at::native_group_norm(input_value, std::nullopt, std::nullopt, N, C, HxW, group, eps);
  }

  if (weight.defined()) {
    const auto padded_weight = padRight(weight, std::nullopt, result0.dim() - 1);
    result0 = result0 * padded_weight;
  }

  if (bias.defined()) {
    const auto padded_bias = padRight(bias, std::nullopt, result0.dim() - 1);
    result0 = result0 + padded_bias;
  }

  return std::make_tuple(std::move(result0), std::move(mean), std::move(rstd));
}

static at::Tensor group_norm_backward_no_weight_bias_batch_rule(
    const at::Tensor & grad_out, std::optional<int64_t> grad_out_bdim,
    const at::Tensor & input, std::optional<int64_t> input_bdim,
    const at::Tensor & mean, std::optional<int64_t> mean_bdim,
    const at::Tensor & rstd, std::optional<int64_t> rstd_bdim,
    int64_t N, int64_t C, int64_t HxW, int64_t group) {
  auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto mean_ = moveBatchDimToFront(mean, mean_bdim);
  auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

  const auto bdim_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  grad_out_ = ensure_has_bdim(grad_out, grad_out_bdim.has_value(), bdim_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
  mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
  rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

  grad_out_ = reshape_dim_into(0, 0, grad_out_); // [B0 * N, C, *]
  input_ = reshape_dim_into(0, 0, input_);       // [B0 * N, C, *]
  mean_ = reshape_dim_into(0, 0, mean_);         // [B0 * N, G]
  rstd_ = reshape_dim_into(0, 0, rstd_);         // [B0 * N, G]

  auto result0 = std::get<0>(native_group_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      mean_.contiguous(),
      rstd_.contiguous(),
      std::nullopt, N * bdim_size, C, HxW, group, {true, false, false}));
  return reshape_dim_outof(0, bdim_size, result0);
}

static std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward_plumbing(
  const Tensor & grad_out, const Tensor & input, const Tensor & mean,
  const Tensor & rstd, const std::optional<Tensor> & weight_opt,
  int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask
) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // plumbing
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "native_group_norm_backward_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();

  if (!areAnyBatchedAtLevel({grad_out, input, mean, rstd, weight_opt}, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::native_group_norm_backward(grad_out, input, mean, rstd, weight_opt, N, C, HxW, group, output_mask);
  }

  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  Tensor weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight.defined()){
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }
  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);
  auto [rstd_value, rstd_bdim] = unwrapTensorAtLevel(rstd, cur_level);

  // results
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  TORCH_INTERNAL_ASSERT(grad_out.dim() > 1);  // group_norm can't operate on 1D tensors so the output will be at least 2D
  if (output_mask[2]) {
    grad_bias = grad_out.transpose(0, 1).sum(range(1, grad_out.dim()));
  }

  if (output_mask[1] && weight.defined()) {
    const auto reshaped_input = reshape_dim_outof(1, group, input);
    const auto normalized_input = (reshaped_input - padRight(mean, std::nullopt, reshaped_input.dim())) * padRight(rstd, std::nullopt, reshaped_input.dim());
    const auto expanded_grad_weight = reshape_dim_into(1, 1, normalized_input) * grad_out;
    grad_weight = expanded_grad_weight.transpose(0, 1).sum(range(1, expanded_grad_weight.dim()));
  }

  if (output_mask[0]) {
    const auto grad_normalized_input = weight.defined() ?
      grad_out * padRight(weight, std::nullopt, grad_out.dim() - 1) : grad_out;
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input, cur_level);

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto tensor = group_norm_backward_no_weight_bias_batch_rule(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        mean_value, mean_bdim,
        rstd_value, rstd_bdim,
        N, C, HxW, group
    );
    grad_input = makeBatched(std::move(tensor), 0, cur_level);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static bool has_same_shape(
    const Tensor& tensor, std::optional<int64_t> tensor_bdim,
    c10::SymIntArrayRef normalized_shape) {
  if (!tensor.defined()) {
    return true;
  }
  if (rankWithoutBatchDim(tensor, tensor_bdim) != static_cast<int64_t>(normalized_shape.size())) {
    return false;
  }
  const auto tensor_shape = tensor.sizes();
  for (const auto i : c10::irange(normalized_shape.size())) {
    auto j = i;
    // (0, 1, 2), 1 -> (0, 2, 3)
    if (tensor_bdim.has_value() && static_cast<int64_t>(i) >= tensor_bdim.value()) {
      j = j + 1;
    }
    if (normalized_shape[i] != tensor_shape[j]) {
      return false;
    }
  }
  return true;
}

static C10_ALWAYS_INLINE void check_same_shape(
    const Tensor& tensor, std::optional<int64_t> tensor_bdim,
    c10::SymIntArrayRef normalized_shape, const std::string& name) {
  TORCH_CHECK(has_same_shape(tensor, tensor_bdim, normalized_shape),
      "Expected ", name, " to be of same shape as normalized_shape, but got ",
      name, " of shape ",
      tensor.sizes(),
      " and normalized_shape = ",
      normalized_shape);
}

// Ugh, hard to deduplicate
static C10_ALWAYS_INLINE void _check_layer_norm_inputs(
    SymIntArrayRef normalized_shape,
    const Tensor& weight, std::optional<int64_t> weight_bdim,
    const Tensor& bias, std::optional<int64_t> bias_bdim) {

  const auto normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  check_same_shape(weight, weight_bdim, normalized_shape, "weight");
  check_same_shape(bias, bias_bdim, normalized_shape, "weight");
}

static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
native_layer_norm_batch_rule(
    const Tensor& input, std::optional<int64_t> input_bdim,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt, std::optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, std::optional<int64_t> bias_bdim,
    double eps) {
  auto input_ = moveBatchDimToFront(input, input_bdim);
  if (!weight_bdim && !bias_bdim) {
    auto [result0, mean, rstd] = at::native_layer_norm_symint(input_, normalized_shape, weight_opt, bias_opt, eps);
    const auto stats_bdim = compute_stat_bdim(input_bdim, mean);
    return std::make_tuple(std::move(result0), 0, std::move(mean), stats_bdim, std::move(rstd), stats_bdim);
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  _check_layer_norm_inputs(normalized_shape, weight, weight_bdim, bias, bias_bdim);

  const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  const auto result = at::native_layer_norm_symint(input_, normalized_shape, std::nullopt, std::nullopt, eps);
  auto [result0, mean, rstd] = result;
  const auto stats_bdim = compute_stat_bdim(input_bdim, mean);

  if (weight.defined()) {
    auto weight_ = moveBatchDimToFront(weight, weight_bdim);
    weight_ = maybePadToLogicalRank(weight_, /*has_bdim*/weight_bdim, input_logical_rank);
    result0 = result0 * weight_;
  }
  if (bias.defined()) {
    const auto result_logical_rank = rankWithoutBatchDim(
        result0,
        input_bdim.has_value() || weight_bdim.has_value() ? std::optional<int64_t>(0) : std::optional<int64_t>(std::nullopt));
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    bias_ = maybePadToLogicalRank(bias_, /*has_bdim*/bias_bdim, result_logical_rank);
    result0 = result0 + bias_;
  }
  return std::make_tuple(result0, 0, mean, stats_bdim, rstd, stats_bdim);
}

static std::tuple<at::Tensor, std::optional<int64_t>> native_layer_norm_backward_no_weight_bias_batch_rule(
    const at::Tensor & grad_out, std::optional<int64_t> grad_out_bdim,
    const at::Tensor & input, std::optional<int64_t> input_bdim,
    at::IntArrayRef normalized_shape,
    const at::Tensor & mean, std::optional<int64_t> mean_bdim,
    const at::Tensor & rstd, std::optional<int64_t> rstd_bdim) {

  if (!grad_out_bdim.has_value() && !input_bdim.has_value() &&
      !mean_bdim.has_value() && !rstd_bdim.has_value()) {
    const auto result = at::native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, std::nullopt, std::nullopt, {true, false, false});
    return std::make_tuple(std::get<0>(result), std::nullopt);
  }

  auto grad_out_ = moveBatchDimToFront(grad_out, grad_out_bdim);
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto mean_ = moveBatchDimToFront(mean, mean_bdim);
  auto rstd_ = moveBatchDimToFront(rstd, rstd_bdim);

  // ensure grad_out / input have bdim.
  const auto bdim_size = get_bdim_size2(grad_out, grad_out_bdim, input, input_bdim);
  grad_out_ = ensure_has_bdim(grad_out_, grad_out_bdim.has_value(), bdim_size);
  input_ = ensure_has_bdim(input_, input_bdim.has_value(), bdim_size);
  mean_ = ensure_has_bdim(mean_, mean_bdim.has_value(), bdim_size);
  rstd_ = ensure_has_bdim(rstd_, rstd_bdim.has_value(), bdim_size);

  auto result = at::native_layer_norm_backward(
      grad_out_.contiguous(),
      input_.contiguous(),
      normalized_shape,
      mean_.contiguous(),
      rstd_.contiguous(),
      std::nullopt, std::nullopt, {true, false, false});

  return std::make_tuple(std::get<0>(result), 0);
}

static std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward_plumbing(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    at::IntArrayRef normalized_shape,
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const std::optional<at::Tensor> & weight_opt,
    const std::optional<at::Tensor> & bias_opt,
    std::array<bool,3> output_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // plumbing
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "native_layer_norm_backward_plumbing");
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  int64_t cur_level = maybe_layer->layerId();
  if (!areAnyBatchedAtLevel({grad_out, input, mean, rstd, weight_opt, bias_opt}, cur_level)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    return at::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd,
        weight_opt, bias_opt, output_mask);
  }
  auto [grad_out_value, grad_out_bdim] = unwrapTensorAtLevel(grad_out, cur_level);
  auto [input_value, input_bdim] = unwrapTensorAtLevel(input, cur_level);
  auto [mean_value, mean_bdim] = unwrapTensorAtLevel(mean, cur_level);
  auto [rstd_value, rstd_bdim] = unwrapTensorAtLevel(rstd, cur_level);
  std::optional<Tensor> weight_value;
  std::optional<int64_t> weight_bdim;
  if (weight.defined()) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight, cur_level);
  }
  std::optional<Tensor> bias_value;
  std::optional<int64_t> bias_bdim;
  if (bias.defined()) {
    std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias, cur_level);
  }

  // results
  Tensor grad_bias;
  Tensor grad_weight;
  Tensor grad_input;

  if (output_mask[2] && bias_value.has_value()) {
    const auto num_front_dims_to_reduce = grad_out.dim() - normalized_shape.size();
    if (num_front_dims_to_reduce == 0) {
      grad_bias = grad_out;
    } else {
      grad_bias = grad_out.sum(range(0, static_cast<int64_t>(num_front_dims_to_reduce)));
    }
  }
  if (output_mask[1] && weight_value.has_value()) {
    // NB: output isn't saved...
    const auto normalized_input = (input - mean) * rstd;
    const auto expanded_grad_weight = normalized_input * grad_out;
    const auto num_front_dims_to_reduce =
        expanded_grad_weight.dim() - normalized_shape.size();
    if (num_front_dims_to_reduce == 0) {
      grad_weight = expanded_grad_weight;
    } else {
      grad_weight = expanded_grad_weight.sum(range(0, static_cast<int64_t>(num_front_dims_to_reduce)));
    }
  }
  if (output_mask[0]) {
    const auto grad_normalized_input = weight.defined() ?
      grad_out * weight : grad_out;
    auto [grad_normalized_input_value, grad_normalized_input_bdim] =
        unwrapTensorAtLevel(grad_normalized_input, cur_level);

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto results = native_layer_norm_backward_no_weight_bias_batch_rule(
        grad_normalized_input_value, grad_normalized_input_bdim,
        input_value, input_bdim,
        normalized_shape,
        mean_value, mean_bdim,
        rstd_value, rstd_bdim);
    grad_input = makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

template <typename F, F Func>
struct NativeBatchNormBatchRuleHelper {
  static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>> apply(
    const Tensor& input, std::optional<int64_t> input_bdim,
    const std::optional<Tensor>& weight_opt, std::optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, std::optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, std::optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, std::optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    return batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
  }
};

template <typename F, F Func>
struct CudnnBatchNormBatchRuleHelper {
  static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>> apply(
    const Tensor& input, std::optional<int64_t> input_bdim,
    const Tensor& weight_opt, std::optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, std::optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, std::optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, std::optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    auto reserve = at::empty({0}, input.options().dtype(kByte));  // in experiments, reserve was never set to anything other than empty by cuda
    auto res = batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
    return std::tuple_cat(res, std::make_tuple(reserve, std::nullopt));
  }
};

template <typename F, F Func>
struct MiopenBatchNormBatchRuleHelper {
  static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>> apply(
    const Tensor& input, std::optional<int64_t> input_bdim,
    const Tensor& weight_opt, std::optional<int64_t> weight_bdim,
    const std::optional<Tensor>& bias_opt, std::optional<int64_t> bias_bdim,
    const std::optional<Tensor>& running_mean_opt, std::optional<int64_t> running_mean_bdim,
    const std::optional<Tensor>& running_var_opt, std::optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps) {
    return batch_norm_batch_rule<F, Func>(
        input, input_bdim, weight_opt, weight_bdim, bias_opt, bias_bdim,
        running_mean_opt, running_mean_bdim, running_var_opt, running_var_bdim, training, momentum, eps);
  }
};

#define NATIVE_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
    NativeBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

#define CUDNN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
   CudnnBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

#define MIOPEN_BATCH_NORM_BATCH_RULE(fn) SINGLE_ARG(\
    MiopenBatchNormBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

template <typename F, F Func>
struct NativeBatchNormBackwardBatchRuleHelper {
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const std::optional<at::Tensor> & weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {

    auto maybe_layer = maybeCurrentDynamicLayer();
    vmap_check_escaped(maybe_layer, "NativeBatchNormBackwardBatchRuleHelper.apply");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    int64_t cur_level = maybe_layer->layerId();

    if (!areAnyBatchedAtLevel({grad_out, input, weight_opt, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt}, cur_level)) {
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      return at::native_batch_norm_backward(grad_out, input, weight_opt,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt,
          training, eps, output_mask);
    }

    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, training, eps, output_mask);
  }
};

template <typename F, F Func>
struct CudnnBatchNormBackwardBatchRuleHelper {
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & input,
    const at::Tensor & grad_out,
    const at::Tensor & weight,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    double eps,
    const at::Tensor & reserve) {

    auto maybe_layer = maybeCurrentDynamicLayer();
    vmap_check_escaped(maybe_layer, "CudnnBatchNormBackwardBatchRuleHelper.apply");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    int64_t cur_level = maybe_layer->layerId();

    if (!areAnyBatchedAtLevel({input, grad_out, weight, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt, reserve}, cur_level)) {
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      return at::cudnn_batch_norm_backward(input, grad_out, weight,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps, reserve);
    }

    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, true, eps, {true, true, true});
  }
};

template <typename F, F Func>
struct MiopenBatchNormBackwardBatchRuleHelper {
  static std::tuple<Tensor,Tensor,Tensor> apply(
    const at::Tensor & input,
    const at::Tensor & grad_out,
    const at::Tensor & weight,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    double eps) {

    auto maybe_layer = maybeCurrentDynamicLayer();
    vmap_check_escaped(maybe_layer, "MiopenBatchNormBackwardBatchRuleHelper.apply");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    int64_t cur_level = maybe_layer->layerId();

    if (!areAnyBatchedAtLevel({input, grad_out, weight, running_mean_opt,
          running_var_opt, save_mean_opt, save_rstd_opt}, cur_level)) {
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      return at::miopen_batch_norm_backward(input, grad_out, weight,
          running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps);
    }

    return batch_norm_backward_plumbing<F, Func>(
        grad_out, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, true, eps, {true, true, true});
  }
};

#define NATIVE_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
    NativeBatchNormBackwardBatchRuleHelper<\
      decltype(&ATEN_FN(fn)),\
      &ATEN_FN(fn)>::apply)

#define CUDNN_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
   CudnnBatchNormBackwardBatchRuleHelper<\
      decltype(&fn),\
      &fn>::apply)

#define MIOPEN_BATCH_NORM_BACKWARD_BATCH_RULE(fn) SINGLE_ARG(\
    MiopenBatchNormBackwardBatchRuleHelper<\
      decltype(&fn),\
      &fn>::apply)

static std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward_wrapper(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const at::Tensor& weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {
    TORCH_INTERNAL_ASSERT(!training);
    auto reserve = at::empty({0}, input.options().dtype(kByte));
    return at::cudnn_batch_norm_backward(input, grad_out, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps, reserve);
  }

static std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward_wrapper(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const at::Tensor& weight_opt,
    const std::optional<at::Tensor> & running_mean_opt,
    const std::optional<at::Tensor> & running_var_opt,
    const std::optional<at::Tensor> & save_mean_opt,
    const std::optional<at::Tensor> & save_rstd_opt,
    bool training,
    double eps,
    std::array<bool,3> output_mask) {
    TORCH_INTERNAL_ASSERT(!training); // this should be ensured by batch_norm_impl
    return at::miopen_batch_norm_backward(input, grad_out, weight_opt, running_mean_opt, running_var_opt, save_mean_opt, save_rstd_opt, eps);
  }

// NB: This is NOT good. In the ideal world, we do NOT want to convert the new legit op back into native_batch_norm
// as native_batch_norm has a problematic schema--it promises it is functional when it is not. However, vmap doesn't
// work with dynamo anyway so we gain some buffer room to do wrong things here. The (reasonable) hope is that we will
// make native_batch_norm composite implicit within a few weeks and we can fix this before vmap works with dynamo.
static std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_batch(
  const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
  Tensor& running_mean, Tensor& running_var, bool train, double momentum, double eps) {
    return at::native_batch_norm(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}

static std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_no_stats_batch(
  const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
  bool train, double momentum, double eps) {
    return at::native_batch_norm(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(native_batch_norm, NATIVE_BATCH_NORM_BATCH_RULE(native_batch_norm));
  VMAP_SUPPORT(cudnn_batch_norm, CUDNN_BATCH_NORM_BATCH_RULE(cudnn_batch_norm));
  VMAP_SUPPORT(miopen_batch_norm, MIOPEN_BATCH_NORM_BATCH_RULE(miopen_batch_norm));
  m.impl("_native_batch_norm_legit", _native_batch_norm_legit_batch);
  m.impl("_native_batch_norm_legit.no_stats", _native_batch_norm_legit_no_stats_batch);
  m.impl("native_batch_norm_backward", NATIVE_BATCH_NORM_BACKWARD_BATCH_RULE(native_batch_norm_backward));
  m.impl("cudnn_batch_norm_backward", CUDNN_BATCH_NORM_BACKWARD_BATCH_RULE(at::functorch::cudnn_batch_norm_backward_wrapper));
  m.impl("miopen_batch_norm_backward", MIOPEN_BATCH_NORM_BACKWARD_BATCH_RULE(at::functorch::miopen_batch_norm_backward_wrapper));
  m.impl("native_group_norm", native_group_norm_plumbing);
  m.impl("native_group_norm_backward", native_group_norm_backward_plumbing);
  VMAP_SUPPORT(native_layer_norm, native_layer_norm_batch_rule);
  m.impl("native_layer_norm_backward", native_layer_norm_backward_plumbing);
}

} // namespace at::functorch
