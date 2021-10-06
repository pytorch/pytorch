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

std::vector<optional<Tensor>> batchIndices(
  ArrayRef<optional<Tensor>> indices,
  ArrayRef<optional<int64_t>> indices_bdims,
  int64_t batch_size,
  optional<int64_t> self_bdim,
  optional<int64_t> values_bdim = nullopt) {
  // There are 3 main cases:
  // 1. self is batched, indices/values are not batched
  // In this case, we just need to augment indices with a None at the front to
  // basically broadcast the indexing across the batch dimension of self.
  //
  // 2. self is not batched, some indices are batched.
  // In this case, we don't need to do anything - indices will automatically
  // broadcast to work with the unbatched self.
  //
  // 3. self is batched, some indices are batched.
  // In this case, we simply need to add an arange that indexes along the first
  // dimension (i.e. the batch dimension). We also need to make sure this
  // broadcasts with the rest of the indices.
  //
  // There is one more case worth mentioning - boolean tensor indices. If we
  // have "batched" boolean tensor indices, that is unrepresentable, as each
  // batch would result in a tensor with different values.
  std::vector<optional<Tensor>> indices_;
  int64_t minIndexDim = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    auto index = indices[i];
    if (index.has_value()) {
      indices_.push_back(moveBatchDimToFront(index.value(), indices_bdims[i]));
      minIndexDim = std::max(minIndexDim, index.value().dim());
      if (index.value().dtype() == kBool && indices_bdims[i].has_value()) {
        throw std::runtime_error("vmap: We do not support batching operators that can support dynamic shape. Attempting to batch over indexing with a boolean mask.");
      }
    } else {
      indices_.push_back(index);
    }
  }

  bool indices_batched = false;
  for (auto idx : indices_bdims) {
    indices_batched = indices_batched || idx.has_value();
  }
  if (!indices_batched && values_bdim.has_value()) {
    minIndexDim += 1;
  }

  if (!indices_batched && self_bdim.has_value()) {
    indices_.insert(indices_.begin(), nullopt);
  } else if (indices_batched && !self_bdim.has_value()) {
    // do nothing
  } else if (indices_batched && (self_bdim.has_value() || values_bdim.has_value())) {
    auto arange_index = at::arange(0, batch_size);
    while (arange_index.dim() < minIndexDim) {
      arange_index = arange_index.unsqueeze(-1);
    }
    indices_.insert(indices_.begin(), arange_index);
  }
  return indices_;
}

std::tuple<Tensor,optional<int64_t>> index_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims) {

  auto self_ = moveBatchDimToFront(self, self_bdim);
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  std::vector<optional<Tensor>> indices_ = batchIndices(indices, indices_bdims, self_.size(0), self_bdim);
  return std::make_tuple(at::index(self_, List<optional<Tensor>>(indices_)), 0);
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

void index_put__batch_rule(
    Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate) {
  if (!self_bdim.has_value()) {
    vmapIncompatibleInplaceError("index_put");
  }
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto values_ = moveBatchDimToFront(values, values_bdim);
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  std::vector<optional<Tensor>> indices_ = batchIndices(indices, indices_bdims, self_.size(0), self_bdim, values_bdim);
  at::index_put_(self_, List<optional<Tensor>>(indices_), values, accumulate);
}

Tensor& index_put__plumbing(Tensor & self, const List<optional<Tensor>> & indices
, const Tensor & values, bool accumulate) {
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
  Tensor values_value;
  optional<int64_t> values_bdim;
  std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
  index_put__batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate);
  return self;
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

std::tuple<Tensor,optional<int64_t>> scatter_value_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value) {
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

  auto result = at::scatter(self_, physical_dim, index_, value);
  // result should have same shape as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

namespace {

template <typename Func>
inline std::tuple<Tensor,optional<int64_t>> scatter_batch_rule(
    Func f,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto src_logical_rank = rankWithoutBatchDim(src, src_bdim);
  auto batch_size = bdim_size(self, self_bdim, index, index_bdim, src, src_bdim);

  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto index_ = moveBatchDimToFront(index, index_bdim);
  auto src_ = moveBatchDimToFront(src, src_bdim);

  if (self_logical_rank == 0) {
    self_ = self_.unsqueeze(-1);
  }
  if (index_logical_rank == 0) {
    index_ = index_.unsqueeze(-1);
  }
  if (src_logical_rank == 0) {
    src_ = src_.unsqueeze(-1);
  }
  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  index_ = ensure_has_bdim(index_, index_bdim.has_value(), batch_size);
  src_ = ensure_has_bdim(src_, src_bdim.has_value(), batch_size);
  auto physical_dim = getPhysicalDim(self_, /*has_batch_dim*/true, dim);

  auto result = f(self_, physical_dim, index_, src_);
  // result should have same shape as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

} // namespace

std::tuple<Tensor,optional<int64_t>> scatter_src_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  return scatter_batch_rule(ATEN_FN2(scatter, src),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
}

std::tuple<Tensor,optional<int64_t>> scatter_add_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim) {
  return scatter_batch_rule(ATEN_FN(scatter_add),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim);
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
  m.impl("index.Tensor", index_plumbing);
  m.impl("index_put_", index_put__plumbing);
  VMAP_SUPPORT("gather", gather_batch_rule);
  VMAP_SUPPORT("gather_backward", gather_backward_batch_rule);
  VMAP_SUPPORT("scatter.value", scatter_value_batch_rule);
  VMAP_SUPPORT("scatter.src", scatter_src_batch_rule);
  VMAP_SUPPORT("scatter_add", scatter_add_batch_rule);
}

}}
