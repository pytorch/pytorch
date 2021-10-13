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

static Tensor reshape_bdim_into_front(
    const Tensor& value,
    optional<int64_t> bdim,
    int64_t batch_size,
    bool is_no_batch_dim_case) {
  auto value_ = ensure_has_bdim(value, bdim.has_value(), batch_size);
  if (!bdim.has_value()) {
    bdim = 0;
  }
  if (is_no_batch_dim_case) {
    return moveBatchDimToFront(value_, bdim);
  }
  return reshape_dim_into(*bdim, 0, value_);
}

// We can't use ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED because the CUDA
// kernel rightfully assumes that indices is contiguous.
std::tuple<Tensor,optional<int64_t>> max_pool2d_with_indices_backward_batch_rule(
    const Tensor& gradOutput, optional<int64_t> gradOutput_bdim,
    const Tensor& input, optional<int64_t> input_bdim,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices, optional<int64_t> indices_bdim) {
  TORCH_INTERNAL_ASSERT(input_bdim.has_value() ^ !indices_bdim.has_value());
  const auto bdim_size = get_bdim_size2(gradOutput, gradOutput_bdim, input, input_bdim);
  const auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);
  bool chw_case = input_logical_rank == 3;

  const auto gradOutput_ = reshape_bdim_into_front(gradOutput, gradOutput_bdim, bdim_size, chw_case);
  const auto input_ = reshape_bdim_into_front(input, input_bdim, bdim_size, chw_case);
  const auto indices_ = reshape_bdim_into_front(indices, indices_bdim, bdim_size, chw_case);

  const auto result = at::max_pool2d_with_indices_backward(
      gradOutput_, input_, kernel_size, stride, padding, dilation, ceil_mode,
      // max_pool2d_with_indices rightfully assumes that indices is contiguous
      indices_.contiguous());

  if (chw_case) {
    return std::make_tuple(std::move(result), 0);
  } else {
    return std::make_tuple(reshape_dim_outof(0, bdim_size, result), 0);
  }
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool2d_with_indices_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  TORCH_INTERNAL_ASSERT(logical_rank == 3 || logical_rank == 4);
  // Tensor[B, C, H, W] -> just call max_pool2d
  if (logical_rank == 3) {
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto result = at::max_pool2d_with_indices(
        self_, kernel_size, stride, padding, dilation, ceil_mode);
    return std::make_tuple(std::move(std::get<0>(result)), 0, std::move(std::get<1>(result)), 0);
  }
  // Tensor[B, N, C, H, W] -> Tensor[B * N, C, H, W]
  auto bdim_size = self.size(*self_bdim);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto result = at::max_pool2d_with_indices(
      self_, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(
      reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  EXISTING_BDIM(_adaptive_avg_pool2d);
  EXISTING_BDIM(avg_pool2d);
  EXISTING_BDIM_ALL_BOXED(avg_pool2d_backward);
  VMAP_SUPPORT("max_pool2d_with_indices", max_pool2d_with_indices_batch_rule);
  VMAP_SUPPORT("max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_batch_rule);
}

}}
