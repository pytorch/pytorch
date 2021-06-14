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

std::tuple<Tensor,optional<int64_t>> adaptive_avg_pool2d_batch_rule(
    const Tensor& tensor, optional<int64_t> batch_dim, IntArrayRef output_size) {
  if (!batch_dim) {
    return std::make_tuple( at::adaptive_avg_pool2d(tensor, output_size), nullopt );
  }
  auto batch_size = tensor.size(*batch_dim);
  auto tensor_ = reshape_dim_into(*batch_dim, 0, tensor);
  auto result = at::adaptive_avg_pool2d(tensor_, output_size);
  return std::make_tuple( reshape_dim_outof(0, batch_size, result), 0 );
}

std::tuple<Tensor,int64_t> max_pool2d_with_indices_backward_batch_rule(
    const Tensor & grad_output, optional<int64_t> grad_output_bdim,
    const Tensor & self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool ceil_mode,
    const Tensor & indices, optional<int64_t> indices_bdim) {
  TORCH_INTERNAL_ASSERT(grad_output_bdim && self_bdim && indices_bdim);

  auto bdim_size = self.size(*self_bdim);
  auto grad_output_ = reshape_dim_into(*grad_output_bdim, 0, grad_output);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto indices_ = reshape_dim_into(*indices_bdim, 0, indices);

  auto result = at::max_pool2d_with_indices_backward(
      grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode,
      indices_);

  result = reshape_dim_outof(0, bdim_size, result);
  return {result, 0};
}

Tensor max_pool2d_with_indices_backward_plumbing(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();

  Tensor grad_output_value;
  optional<int64_t> grad_output_bdim;
  std::tie(grad_output_value, grad_output_bdim) = unwrapTensorAtLevel(grad_output, cur_level);
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  Tensor indices_value;
  optional<int64_t> indices_bdim;
  std::tie(indices_value, indices_bdim) = unwrapTensorAtLevel(indices, cur_level);

  if (self_bdim && grad_output_bdim && indices_bdim) {
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto result = max_pool2d_with_indices_backward_batch_rule(
        grad_output_value, grad_output_bdim,
        self_value, self_bdim,
        kernel_size, stride, padding, dilation, ceil_mode,
        indices_value, indices_bdim);
    return makeBatched(std::get<0>(result), std::get<1>(result), cur_level);
  }

  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::max_pool2d_with_indices_backward", "");
  return slow_fallback<Tensor>(op, { grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices });
}


TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("adaptive_avg_pool2d", adaptive_avg_pool2d_batch_rule);
  m.impl("max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_plumbing);
}

}}
