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

Tensor ensure_has_bdim(const Tensor& tensor, bool has_bdim, int64_t batch_size) {
  if (has_bdim) {
    return tensor;
  }
  const auto sizes = tensor.sizes();
  DimVector expanded_shape;
  expanded_shape.reserve(sizes.size());
  expanded_shape.emplace_back(batch_size);
  expanded_shape.insert(expanded_shape.end(), sizes.begin(), sizes.end());
  return tensor.expand(expanded_shape);
}

int64_t bdim_size(
    const Tensor& a, optional<int64_t> a_bdim,
    const Tensor& b, optional<int64_t> b_bdim,
    const Tensor& c, optional<int64_t> c_bdim) {
  if (a_bdim) {
    return a.size(*a_bdim);
  }
  if (b_bdim) {
    return b.size(*b_bdim);
  }
  if (c_bdim) {
    return c.size(*c_bdim);
  }
  TORCH_INTERNAL_ASSERT(false);
}

int64_t bdim_size(
    const Tensor& a, optional<int64_t> a_bdim,
    const Tensor& b, optional<int64_t> b_bdim) {
  if (a_bdim) {
    return a.size(*a_bdim);
  }
  if (b_bdim) {
    return b.size(*b_bdim);
  }
  TORCH_INTERNAL_ASSERT(false);
}

std::tuple<Tensor,optional<int64_t>> gather_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto batch_size = bdim_size(self, self_bdim, index, index_bdim);

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  auto result = at::gather(self_, physical_dim, index_, sparse_grad);
  // result should have same shape as index
  if (index_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

std::tuple<Tensor,optional<int64_t>> gather_backward_batch_rule(
    const Tensor& grad, optional<int64_t> grad_bdim,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  auto batch_size = bdim_size(grad, grad_bdim, self, self_bdim, index, index_bdim);
  auto grad_ = moveBatchDimToFront(grad, grad_bdim);
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);

  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto grad_logical_rank = rankWithoutBatchDim(grad, grad_bdim);

  if (grad_logical_rank == 0) {
    grad_ = grad_.unsqueeze(-1);
  }
  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  grad_ = ensure_has_bdim(grad_, grad_bdim.has_value(), batch_size);
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);

  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);
  auto result = at::gather_backward(grad_, self_, physical_dim, index_, sparse_grad);
  // result should has same shape as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
    // m.impl("index.Tensor", index_plumbing);
  VMAP_SUPPORT("gather", gather_batch_rule);
  VMAP_SUPPORT("gather_backward", gather_backward_batch_rule);
}

}}
