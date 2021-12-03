// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

static optional<Tensor> maybe_flatten(
    const optional<Tensor>& tensor, optional<int64_t> tensor_bdim) {
  if (!tensor.has_value()) {
    return nullopt;
  }
  TORCH_INTERNAL_ASSERT(tensor_bdim.has_value());
  return reshape_dim_into(*tensor_bdim, 0, *tensor);
}

std::tuple<at::Tensor,optional<int64_t>>
batch_norm_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    const c10::optional<Tensor>& weight, optional<int64_t> weight_bdim,
    const c10::optional<Tensor>& bias, optional<int64_t> bias_bdim,
    const c10::optional<Tensor>& running_mean, optional<int64_t> running_mean_bdim,
    const c10::optional<Tensor>& running_var, optional<int64_t> running_var_bdim,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  TORCH_INTERNAL_ASSERT(input_bdim.has_value());
  auto batch_size = input.size(*input_bdim);
  auto input_ = reshape_dim_into(*input_bdim, /*channels dim*/1, input);

  auto weight_ = maybe_flatten(weight, weight_bdim);
  auto bias_ = maybe_flatten(bias, bias_bdim);
  auto running_mean_ = maybe_flatten(running_mean, running_mean_bdim);
  auto running_var_ = maybe_flatten(running_var, running_var_bdim);

  auto result = at::batch_norm(
      input_, weight_, bias_, running_mean_, running_var_,
      training, momentum, eps, cudnn_enabled);
  return std::make_tuple(reshape_dim_outof(1, batch_size, result), 1);
}

Tensor batch_norm_plumbing(
    const Tensor& input,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);

  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
    std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(*weight, cur_level);
  }

  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
    std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(*bias, cur_level);
  }

  optional<Tensor> running_mean_value;
  optional<int64_t> running_mean_bdim;
  if (running_mean) {
    std::tie(running_mean_value, running_mean_bdim) = unwrapTensorAtLevel(*running_mean, cur_level);
  }

  optional<Tensor> running_var_value;
  optional<int64_t> running_var_bdim;
  if (running_var) {
    std::tie(running_var_value, running_var_bdim) = unwrapTensorAtLevel(*running_var, cur_level);
  }

  if (input_bdim &&
      (!weight.has_value() || weight_bdim) &&
      (!bias.has_value() || bias_bdim) &&
      (!running_mean.has_value() || running_mean_bdim) &&
      (!running_var.has_value() || running_var_bdim)) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto results = batch_norm_batch_rule(
        input_value, input_bdim,
        weight_value, weight_bdim,
        bias_value, bias_bdim,
        running_mean_value, running_mean_bdim,
        running_var_value, running_var_bdim,
        training, momentum, eps, cudnn_enabled);
    return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::batch_norm", "");
  return slow_fallback<Tensor>(op, {
      input, weight, bias, running_mean, running_var,
      training, momentum, eps, cudnn_enabled});
}


std::tuple<Tensor,int64_t,Tensor,int64_t,Tensor,int64_t>
native_group_norm_input_batch_rule(
    const Tensor & input, int64_t input_bdim,
    const c10::optional<Tensor> & weight,
    const c10::optional<Tensor> & bias, int64_t N, int64_t C,
    int64_t HxW, int64_t group, double eps) {
  auto bdim_size = input.size(input_bdim);
  auto input_ = reshape_dim_into(input_bdim, 0, input);
  auto result = at::native_group_norm(input_, weight, bias, N * bdim_size, C, HxW, group, eps);
  return std::make_tuple(
      reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<2>(result)), 0);
}

std::tuple<Tensor,Tensor,Tensor> native_group_norm_plumbing(
    const Tensor & input, const c10::optional<Tensor> & weight,
    const c10::optional<Tensor> & bias, int64_t N, int64_t C,
    int64_t HxW, int64_t group, double eps) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor input_value;
  optional<int64_t> input_bdim;
  std::tie(input_value, input_bdim) = unwrapTensorAtLevel(input, cur_level);
  optional<Tensor> weight_value;
  optional<int64_t> weight_bdim;
  if (weight) {
      std::tie(weight_value, weight_bdim) = unwrapTensorAtLevel(weight.value(), cur_level);
  }
  optional<Tensor> bias_value;
  optional<int64_t> bias_bdim;
  if (bias) {
      std::tie(bias_value, bias_bdim) = unwrapTensorAtLevel(bias.value(), cur_level);
  }

  if (input_bdim && !weight_bdim && !bias_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto result = native_group_norm_input_batch_rule(
        input_value, *input_bdim, weight_value, bias_value,
        N, C, HxW, group, eps);
    return std::make_tuple(
        makeBatched(std::get<0>(result), std::get<1>(result), cur_level),
        makeBatched(std::get<2>(result), std::get<3>(result), cur_level),
        makeBatched(std::get<4>(result), std::get<5>(result), cur_level));
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::native_group_norm", "");
  return slow_fallback<Tensor,Tensor,Tensor>(op, { input, weight, bias, N, C, HxW, group, eps });
}

C10_ALWAYS_INLINE bool has_same_shape(
    const Tensor& tensor, optional<int64_t> tensor_bdim,
    IntArrayRef normalized_shape) {
  if (!tensor.defined()) {
    return true;
  }
  if (rankWithoutBatchDim(tensor, tensor_bdim) != normalized_shape.size()) {
    return false;
  }
  const auto tensor_shape = tensor.sizes();
  for (const auto i : c10::irange(normalized_shape.size())) {
    auto j = i;
    // (0, 1, 2), 1 -> (0, 2, 3)
    if (tensor_bdim.has_value() && (int64_t)i >= tensor_bdim.value()) {
      j = j + 1;
    }
    if (normalized_shape[i] != tensor_shape[j]) {
      return false;
    }
  }
  return true;
}

C10_ALWAYS_INLINE void check_same_shape(
    const Tensor& tensor, optional<int64_t> tensor_bdim,
    IntArrayRef normalized_shape, const std::string& name) {
  TORCH_CHECK(has_same_shape(tensor, tensor_bdim, normalized_shape),
      "Expected ", name, " to be of same shape as normalized_shape, but got ",
      name, " of shape ",
      tensor.sizes(),
      " and normalized_shape = ",
      normalized_shape);
}

// Ugh, hard to deduplicate
C10_ALWAYS_INLINE void _check_layer_norm_inputs(
    IntArrayRef normalized_shape,
    const Tensor& weight, optional<int64_t> weight_bdim,
    const Tensor& bias, optional<int64_t> bias_bdim) {

  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  check_same_shape(weight, weight_bdim, normalized_shape, "weight");
  check_same_shape(bias, bias_bdim, normalized_shape, "weight");
}

static bool is_empty_tensor(const Tensor& tensor) {
  const auto shape = tensor.sizes();
  return shape.size() == 1 && shape[0] == 0;
}

static optional<int64_t> compute_stat_bdim(
    optional<int64_t> input_bdim,
    const Tensor& stat) {
  // There's a weird case where mean, rstd can both have shape (0,).
  // It's possible that this is a bug on the PyTorch side.
  // When that happens we don't want to return a BatchedTensor.
  if (input_bdim.has_value() && !is_empty_tensor(stat)) {
    return 0;
  }
  return nullopt;
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>,Tensor,optional<int64_t>>
native_layer_norm_batch_rule(
    const Tensor& input, optional<int64_t> input_bdim,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt, optional<int64_t> weight_bdim,
    const c10::optional<Tensor>& bias_opt, optional<int64_t> bias_bdim,
    double eps) {
  auto input_ = moveBatchDimToFront(input, input_bdim);
  if (!weight_bdim && !bias_bdim) {
    const auto result = at::native_layer_norm(input_, normalized_shape, weight_opt, bias_opt, eps);
    const auto mean = std::get<1>(result);
    const auto rstd = std::get<2>(result);
    const auto stats_bdim = compute_stat_bdim(input_bdim, mean);
    return std::make_tuple(std::get<0>(result), 0, mean, stats_bdim, rstd, stats_bdim);
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  _check_layer_norm_inputs(normalized_shape, weight, weight_bdim, bias, bias_bdim);

  const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  const auto result = at::native_layer_norm(input_, normalized_shape, nullopt, nullopt, eps);
  auto result0 = std::get<0>(result);
  const auto mean = std::get<1>(result);
  const auto rstd = std::get<2>(result);
  const auto stats_bdim = compute_stat_bdim(input_bdim, mean);

  if (weight.defined()) {
    auto weight_ = moveBatchDimToFront(weight, weight_bdim);
    weight_ = maybePadToLogicalRank(weight_, /*has_bdim*/weight_bdim, input_logical_rank);
    result0 = result0 * weight_;
  }
  if (bias.defined()) {
    const auto result_logical_rank = rankWithoutBatchDim(
        result0,
        input_bdim.has_value() || weight_bdim.has_value() ? optional<int64_t>(0) : optional<int64_t>(nullopt));
    auto bias_ = moveBatchDimToFront(bias, bias_bdim);
    bias_ = maybePadToLogicalRank(bias_, /*has_bdim*/bias_bdim, result_logical_rank);
    result0 = result0 + bias_;
  }
  return std::make_tuple(result0, 0, mean, stats_bdim, rstd, stats_bdim);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("batch_norm", batch_norm_plumbing);
  m.impl("native_group_norm", native_group_norm_plumbing);
  VMAP_SUPPORT("native_layer_norm", native_layer_norm_batch_rule);
}

}}
