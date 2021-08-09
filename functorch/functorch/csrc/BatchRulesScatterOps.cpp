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

std::tuple<Tensor,optional<int64_t>> gather_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  auto physical_dim = dim;
  if (self_logical_rank == 0) {
    TORCH_CHECK(dim == 0 || dim == -1,
        "Dimension out of range (expected to be in range of [-1, 0], but got ", dim, ")");
    physical_dim = -1;
  } else {
    physical_dim = getPhysicalDim(self_, self_bdim.has_value(), dim);
  }
  if (!self_bdim) {
    TORCH_INTERNAL_ASSERT(index_bdim);
    DimVector expanded_shape(self_.sizes().begin(), self_.sizes().end());
    expanded_shape.insert(expanded_shape.begin(), index_.size(0));
    self_ = self_.expand(expanded_shape);
    physical_dim += 1;
  }
  if (!index_bdim) {
    TORCH_INTERNAL_ASSERT(self_bdim);
    DimVector expanded_shape(index_.sizes().begin(), index_.sizes().end());
    expanded_shape.insert(expanded_shape.begin(), self_.size(0));
    index_ = index_.expand(expanded_shape);
  }
  // Ridiculous special case for scalar tensors
  if (self_logical_rank == 1 && index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
    auto result = at::gather(self_, physical_dim, index_, sparse_grad).squeeze(-1);
    return std::make_tuple(result, 0);
  }
  if (self_logical_rank == 0 && index_logical_rank == 1) {
    self_ = self_.unsqueeze(-1);
  }

  return std::make_tuple(at::gather(self_, physical_dim, index_, sparse_grad), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
    // m.impl("index.Tensor", index_plumbing);
  VMAP_SUPPORT("gather", gather_batch_rule);
}

}}
