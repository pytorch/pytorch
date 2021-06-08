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

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("adaptive_avg_pool2d", adaptive_avg_pool2d_batch_rule);
}

}}
