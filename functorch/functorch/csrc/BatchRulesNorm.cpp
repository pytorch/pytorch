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
  return { reshape_dim_outof(1, batch_size, result), 1 };
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

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  m.impl("batch_norm", batch_norm_plumbing);
}

}}
