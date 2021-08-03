// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <iostream>
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>


namespace at { namespace functorch {

std::tuple<Tensor,optional<int64_t>> index_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  std::vector<optional<Tensor>> indices_;
  for (int idx=0; idx < indices.size(); idx++) {
      if (indices_bdims[idx].has_value()) {
          indices_.push_back(moveBatchDimToFront(*indices[idx], indices_bdims[idx]));
      } else {
          indices_.push_back(indices[idx]);
      }
  }
  auto result = at::index(self_, List<optional<Tensor>>(indices_));
  return std::make_tuple(result, 0);
}

Tensor index_plumbing(const Tensor & self, const List<optional<Tensor>> & indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  Tensor self_value;
  optional<int64_t> self_bdim;
  std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
  std::vector<optional<Tensor>> indices_value;
  std::vector<optional<int64_t>> indices_bdims;
  for (const auto&& indRef : indices) {
      optional<Tensor> ind = indRef;
      optional<Tensor> index;
      optional<int64_t> index_bdim;
      if (ind.has_value()) {
        std::tie(index, index_bdim) = unwrapTensorAtLevel(ind.value(), cur_level);
      }
    indices_value.push_back(index);
    indices_bdims.push_back(index_bdim);
  }
  auto results = index_batch_rule(self_value, self_bdim, indices_value, indices_bdims);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
    // m.impl("index.Tensor", index_plumbing);
}

}}
