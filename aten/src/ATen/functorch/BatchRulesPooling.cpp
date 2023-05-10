// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

template <typename Func>
std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool_with_indices_batch_rule_helper(
  const Tensor& self, optional<int64_t> self_bdim,
  IntArrayRef kernel_size, IntArrayRef stride,
  IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, int64_t n, Func pooling_fn) {

  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  TORCH_INTERNAL_ASSERT(logical_rank == n + 1 || logical_rank == n + 2);
  // Tensor[B, logical_rank...] -> just call max_poolnd
  if (logical_rank == n + 1) {
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto result = pooling_fn(
        self_, kernel_size, stride, padding, dilation, ceil_mode);
    return std::make_tuple(std::move(std::get<0>(result)), 0, std::move(std::get<1>(result)), 0);
  }
  // Tensor[B, N, logical_rank...] -> Tensor[B * N, logical_rank...]
  auto bdim_size = self.size(*self_bdim);
  auto self_ = reshape_dim_into(*self_bdim, 0, self);
  auto result = pooling_fn(
      self_, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(
      reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0);
}

static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool3d_with_indices_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 3, at::max_pool3d_with_indices);
}

static std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
max_pool2d_with_indices_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 2, at::max_pool2d_with_indices);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  EXISTING_BDIM(_adaptive_avg_pool2d);
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool2d_backward);
  EXISTING_BDIM(_adaptive_avg_pool3d);
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool3d_backward);
  EXISTING_BDIM(avg_pool2d);
  EXISTING_BDIM(avg_pool3d);
  EXISTING_BDIM_ALL_BOXED(avg_pool2d_backward);
  EXISTING_BDIM_ALL_BOXED(avg_pool3d_backward);
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool2d);
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool3d);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, adaptive_max_pool2d_backward, 2);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, adaptive_max_pool3d_backward, 2);
  VMAP_SUPPORT(max_pool2d_with_indices, max_pool2d_with_indices_batch_rule);
  VMAP_SUPPORT(max_pool3d_with_indices, max_pool3d_with_indices_batch_rule);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, max_pool2d_with_indices_backward, 2);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, max_pool3d_with_indices_backward, 2);
}

}}
