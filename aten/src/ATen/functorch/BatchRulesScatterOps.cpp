// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <iostream>
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>


namespace at { namespace functorch {

static bool any_has_value(ArrayRef<optional<int64_t>> bdims) {
  for (const auto& bdim : bdims) {
    if (bdim.has_value()) {
      return true;
    }
  }
  return false;
}

static int64_t get_num_leading_nones(ArrayRef<optional<Tensor>> indices) {
  int64_t result = 0;
  for (const auto& idx : indices) {
    if (!idx.has_value() || !idx->defined()) {
      result++;
    } else {
      return result;
    }
  }
  return result;
}

static int64_t get_max_index_logical_dim(
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims) {
  int64_t max_logical_dim = -1;
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  TORCH_INTERNAL_ASSERT(indices.size() > 0);
  for (const auto i : c10::irange(0, indices.size())) {
    const auto& maybe_tensor = indices[i];
    if (!maybe_tensor.has_value() || !maybe_tensor->defined()) {
      continue;
    }
    auto logical_dim = rankWithoutBatchDim(maybe_tensor.value(), indices_bdims[i]);
    max_logical_dim = std::max(logical_dim, max_logical_dim);
  }
  return max_logical_dim;
}

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
  // In all three cases, depending on if advanced indices are adjacent we will
  // have to permute the output.
  // See NOTE: [advanced indexing (index.Tensor) batch rule] for more details
  //
  // There is one more case worth mentioning - boolean tensor indices. If we
  // have "batched" boolean tensor indices, that is unrepresentable, as each
  // batch would result in a tensor with different values.
  std::vector<optional<Tensor>> indices_;

  int64_t maxLogicalRank = get_max_index_logical_dim(indices, indices_bdims);
  bool indices_batched = any_has_value(indices_bdims);

  for (size_t i = 0; i < indices.size(); i++) {
    auto index = indices[i];
    if (index.has_value() && index->numel() != 0) {
      const auto idx_bdim = indices_bdims[i];
      indices_.emplace_back(maybePadToLogicalRank(moveBatchDimToFront(index.value(), idx_bdim), idx_bdim, maxLogicalRank));
      if (index.value().dtype() == kBool && indices_bdims[i].has_value()) {
        throw std::runtime_error("vmap: We do not support batching operators that can support dynamic shape. Attempting to batch over indexing with a boolean mask.");
      }
    } else {
      indices_.push_back(index);
    }
  }

  auto maxIndexDim = maxLogicalRank;
  if (indices_batched || values_bdim.has_value()) {
    maxIndexDim += 1;
  }

  if (!indices_batched && self_bdim.has_value()) {
    indices_.insert(indices_.begin(), nullopt);
  } else if (indices_batched && !self_bdim.has_value()) {
    // do nothing
  } else if (indices_batched && (self_bdim.has_value() || values_bdim.has_value())) {
    auto arange_index = at::arange(0, batch_size);
    while (arange_index.dim() < maxIndexDim) {
      arange_index = arange_index.unsqueeze(-1);
    }
    // TODO: this is O(N)
    indices_.insert(indices_.begin(), arange_index);
  }
  return indices_;
}

// Define an "advanced index" to be a selection object that is
// a non-trivial Tensor (i.e. it does not represent :).
static bool is_advanced_index(const optional<Tensor>& idx) {
  if (!idx.has_value()) {
    return false;
  }
  if (!idx->defined()) {
    return false;
  }
  return true;
}

// See NOTE: [advanced indices adjacent] for definition
static bool are_advanced_indices_adjacent(ArrayRef<optional<Tensor>> indices) {
  int64_t num_advanced_indices_regions = 0;
  bool in_advanced_indices_region = false;
  for (const auto& idx : indices) {
    if (!in_advanced_indices_region && is_advanced_index(idx)) {
      num_advanced_indices_regions++;
      in_advanced_indices_region = true;
      continue;
    }
    if (in_advanced_indices_region && !is_advanced_index(idx)) {
      in_advanced_indices_region = false;
      continue;
    }
  }
  return num_advanced_indices_regions <= 1;
}

// Given a Tensor[B, <first_region>, <second_region>, ...]
// Swaps the regions to produce Tensor[B, <second_region>, <first_region>, ...]
//
// Concretely speaking, given
// - tensor: Tensor[B, 2, 3, 4, 5, 6, 7, 8]
// - first_region_size: 2
// - second_region_size: 3
// Produces:
// - result: Tensor[B, 4, 5, 6, 2, 3, 7, 8]
//                     -------  ----
//                     region2  region1
static Tensor swap_regions(const Tensor& tensor, int64_t first_region_size, int64_t second_region_size) {
  VmapDimVector permutation(tensor.dim(), 0);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::rotate(
      permutation.begin() + 1,
      permutation.begin() + 1 + first_region_size,
      permutation.begin() + 1 + first_region_size + second_region_size);
  return tensor.permute(permutation);
}

std::tuple<Tensor,optional<int64_t>> index_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims) {

  // NOTE: [advanced indexing (index.Tensor) batch rule]
  //
  // This is a three step procedure:
  // 1. batch `indices`. Depends on self_bdim and indices_bdim.
  // 2. call at::index
  // 3. (maybe) reorder the dimensions in the result.
  // Why is step 3 necessary? Let's take a detour first.
  //
  // NOTE: [advanced indices adjacent]
  // Definition: In a list of optional<Tensor> indices,
  // we say that "advanced indices are adjacent" if ALL advanced indices are
  // not separated by a None (slice).
  //
  // So, for example,
  // [:, :, (0, 1), (0, 1), :] -> True
  // [:, (0, 1), :, (0, 1), :] -> False, the advanced indices are separated by a slice
  //
  // See https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
  // for more details.
  //
  // NOTE: [Why is step 3 necessary?]
  //
  // In the original self[*indices] expression,
  // depending on whether or not the "advanced indices inside `indices` are
  // adjacent", something different happens.
  //
  // For example:
  // - self: Tensor[4, 5, 6, 7]
  // - indices: [:, (0, 1), (0, 1), :] (advanced indices are adjacent)
  // - self[*indices]: Tensor[4, 2, 7]
  // If advanced indices are adjacent, you get the output you would expect.
  // (0, 1), (0, 1) says "please index these two dimensions at (0, 0) and (1, 1)
  // to produce two elements".
  //
  // If advanced indices are not adjacent, it is ambiguous to where the new
  // dimension of size 2 should go. The numpy spec says it should go at the very
  // front of the Tensor.
  //
  // - self: Tensor[4, 5, 6, 7]
  // - indices: [:, (0, 1), :, (0, 1)] (advanced indices not adjacent)
  // - self[*indices]: Tensor[2, 4, 6]
  //
  // Now, this leads to some weird interactions with vmap.
  // The indices might originally have adjacent advanced indices, but after
  // batching them with "batchIndices", they may no longer be adjacent!
  // - indices: [:, (0, 1), (0, 1)]
  // - batched_indices (for example): [(0, 1), :, (0, 1), (0, 1)]
  // This leads to the dimension of size 2 appearing somewhere else.
  //
  // There are a couple of different cases that we walk through in the code below.
  //
  // Background reading for why we care about if the advanced indices are adjacent:
  // https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
  auto self_ = moveBatchDimToFront(self, self_bdim);
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());
  bool advanced_indices_are_adjacent = are_advanced_indices_adjacent(indices);

  // Step 1
  const auto batched_indices = batchIndices(indices, indices_bdims, self_.size(0), self_bdim);
  auto num_leading_nones = get_num_leading_nones(indices);
  auto max_index_dim = get_max_index_logical_dim(indices, indices_bdims);

  // Step 2
  auto res = at::index(self_, List<optional<Tensor>>(batched_indices));

  // Step 3: There are three cases (these match the cases outlined in batchIndices)
  bool self_batched = self_bdim.has_value();
  bool indices_batched = any_has_value(indices_bdims);

  TORCH_INTERNAL_ASSERT(self_batched || indices_batched, "Requires at least one batched to get here");

  // Case 1
  if (self_batched && !indices_batched) {
    if (advanced_indices_are_adjacent) {
      // self: Tensor[B, 5, 6, 7, 8]
      // indices: [:, Tensor[2, 2], Tensor[2, 2], :]
      // batched_indices: [:, :, Tensor[2, 2], Tensor[2, 2], :]
      // res: Tensor[B, 5, 2, 2, 8]
      return std::make_tuple(res, 0);
    } else {
      // self: Tensor[B, 5, 6, 7]
      // indices: [Tensor[2, 2], :, Tensor[2, 2]]
      // batched_indices: [:, Tensor[2, 2], :, Tensor[2, 2]]
      // res: Tensor[2, 2, B, 6]
      return std::make_tuple(res, max_index_dim);
    }
  }

  // Case 2
  if (!self_batched && indices_batched) {
    if (advanced_indices_are_adjacent) {
      // self: Tensor[5, 6, 7, 8]
      // indices: [:, :, Tensor[B, 2, 2], Tensor[2, 2]]
      // batched_indices: indices (no change)
      // res: Tensor[5, 6, B, 2, 2]
      return std::make_tuple(res, num_leading_nones);
    } else {
      // self: Tensor[5, 6, 7, 8, 9]
      // indices: [:, :, Tensor[B, 2, 2], :, Tensor[2, 2]]
      // batched_indices: indices (no change)
      // res: Tensor[B, 2, 2, 5, 6, 8]
      return std::make_tuple(res, 0);
    }
  }

  // Case 3: self_batched and indices_batched
  TORCH_INTERNAL_ASSERT(self_batched && indices_batched);
  if (!advanced_indices_are_adjacent) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [:, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // batched_indices: [arange(B).expand(B, 2, 2), :, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // res: Tensor[B, 2, 2, 5, 7]
    return std::make_tuple(res, 0);
  }
  // In other words, in batched_indices, advanced indices are adjacent
  if (num_leading_nones == 0) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // batched_indices: [arange(B).expand(B, 2, 2), Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // res: Tensor[B, 2, 2, 7, 8]
    return std::make_tuple(res, 0);
  }
  // This is the tricky case. In indices, advanced indices are adjacent.
  // In batched_indices, advanced indices are no longer adjacent
  //
  // self: Tensor[B, 5, 6, 7, 8, 9]
  // indices: [:, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // batched_indices: [arange(B).expand(B, 2, 3), :, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // res: Tensor[B, 2, 3, 5, 6, 9]
  // expected: Tensor[B, 5, 6, 2, 3, 9]
  //
  // The resolution is to move dims around until we get the right shape.
  // The result is set up as [B, <maxIndexDim>, <leading_nones>, ...]
  // we just have to move the <leading_nones> to before the <maxIndexDim> to produce
  // [B, <leading_nones>, <maxIndexDim>, ...]
  return std::make_tuple(swap_regions(res, max_index_dim, num_leading_nones), 0);
}

// plumbing done since we don't support List<optional<Tensor>> in codegen
Tensor index_plumbing(const Tensor & self, const List<optional<Tensor>> & indices
) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level)) {
    return at::index(self, indices);
  }
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

namespace {
  // Code is mostly duplicated from
  // https://github.com/pytorch/pytorch/blob/fb0e27d38a8fdab4e1c14d6378c9e41cb30fd6a3
  // /aten/src/ATen/native/TensorAdvancedIndexing.cpp#L294-L312
  VmapDimVector compute_indexed_shape(const Tensor &src, TensorList indices_list)
  {
    int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
    IntArrayRef replacement_shape;
    for (const auto dim : c10::irange(indices_list.size())) {
      if (!indices_list[dim].defined()) {
        if (dims_indexed == 0) {
          dims_before++;
        } else {
          dims_after++;
        }
      } else {
        dims_indexed++;
        replacement_shape = indices_list[dim].sizes();
      }
    }

    // Replace indexed dimensions in src with stride 0 and the size of the result tensor.
    // The offset in these dimensions is computed by the kernel using the index tensor's
    // values and the stride of src. The new shape is not meaningful. It's used to make
    // the shape compatible with the result tensor.
    auto shape = VmapDimVector(src.sizes());
    int64_t end = dims_before + dims_indexed;
    shape.erase(shape.begin() + dims_before, shape.begin() + end);
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
    return shape;
  }

  // Code is mostly duplicated from
  // https://github.com/pytorch/pytorch/blob/fb0e27d38a8fdab4e1c14d6378c9e41cb30fd6a3
  // /aten/src/ATen/native/TensorAdvancedIndexing.cpp#L379-L405
  VmapDimVector get_indexed_shape(Tensor self, const torch::List<c10::optional<at::Tensor>> &orig)
  {
    at::native::checkIndexTensorTypes(orig);
    // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
    auto indices = at::native::expandTensors(self, orig);
    // next broadcast all index tensors together
    try {
      indices = at::expand_outplace(indices);
    } catch (std::exception &e) {
      TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                               " with shapes ");
    }
    // add missing null Tensors so that it matches self.dim()
    while (indices.size() < static_cast<size_t>(self.dim())) {
      indices.emplace_back();
    }
    // if the non-null indices are not all adjacent, transpose self and indices
    // together so that they're adjacent at the front
    if (!at::native::hasContiguousSubspace(indices)) {
      std::tie(self, indices) = at::native::transposeToFront(self, indices);
    }
    return compute_indexed_shape(self, indices);
  }

  std::tuple<Tensor, std::vector<optional<Tensor>>, Tensor>
  index_put_batch_rule_helper(const Tensor &self,
                              optional<int64_t> self_bdim,
                              ArrayRef<optional<Tensor>> indices,
                              ArrayRef<optional<int64_t>> indices_bdims,
                              const Tensor &values,
                              optional<int64_t> values_bdim,
                              optional<int64_t> opt_batch_size = {}) {

    Tensor self_ = moveBatchDimToFront(self, self_bdim);
    Tensor values_ = moveBatchDimToFront(values, values_bdim);
    // for inplace variants `index_put_` and `_index_put_impl_` we find the batch_size
    // here while for `index_put` does it outside of this function.
    const auto batch_size = opt_batch_size ? opt_batch_size.value() : self_.size(0);
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
    values_ = ensure_has_bdim(values_, values_bdim.has_value(), batch_size);
    TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());

    // we've already made sure that self has bdim at 0.
    const auto indices_ = batchIndices(indices, indices_bdims, batch_size, /*self_bdim=*/0, values_bdim);

    auto indexed_shape = get_indexed_shape(self_, List<optional<Tensor>>(indices_));

    // handle broadcasting support for values
    // Eg. Given `indexed_shape.size()` is 5 and
    // shape of `values` is (N, 2, 3), then following block
    // will reshape `values` to (N, 1, 1, 2, 3).
    if ( (int64_t) indexed_shape.size() > values_.dim()) {
      auto values_sizes = values_.sizes();

      // number of unit dims (for broadcasting value to indexed_shape)
      auto n_unit_dims = indexed_shape.size() - values_sizes.size();
      VmapDimVector new_values_shape(values_sizes.size() + n_unit_dims);

      // add the batch-dim
      new_values_shape[0] = batch_size;

      // insert the unit dims for broadcasting.
      for (const auto idx : c10::irange(n_unit_dims)) {
        // since batch-dim is already be filled.
        new_values_shape[idx + 1] = 1;
      }
      for (const auto idx: c10::irange(1, values_sizes.size())) {
        // since batch and unit dims are already be filled.
        new_values_shape[idx + n_unit_dims] = values_sizes[idx];
      }
      values_ = values_.view(new_values_shape);
    }

    return std::make_tuple(self_, indices_, values_);
  }

  auto unpackSelfAndIndicesAndValuesAtCurrentLevel(const Tensor &self,
                                                   const List<optional<Tensor>> &indices,
                                                   const Tensor &values, int64_t cur_level)
  {
    Tensor self_value;
    optional<int64_t> self_bdim;
    std::tie(self_value, self_bdim) = unwrapTensorAtLevel(self, cur_level);
    std::vector<optional<Tensor>> indices_value;
    std::vector<optional<int64_t>> indices_bdims;
    for (const auto &&indRef : indices)
    {
      optional<Tensor> ind = indRef;
      optional<Tensor> index;
      optional<int64_t> index_bdim;
      if (ind.has_value())
      {
        std::tie(index, index_bdim) = unwrapTensorAtLevel(ind.value(), cur_level);
      }
      indices_value.push_back(index);
      indices_bdims.push_back(index_bdim);
    }
    Tensor values_value;
    optional<int64_t> values_bdim;
    std::tie(values_value, values_bdim) = unwrapTensorAtLevel(values, cur_level);
    return std::make_tuple(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim);
  }

}  // namespace

void index_put__batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate) {
  if (!self_bdim.has_value()) {
    vmapIncompatibleInplaceError("index_put_");
  }
  Tensor self_, values_;
  std::vector<optional<Tensor>> indices_;
  std::tie(self_, indices_, values_) = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim);
  at::index_put_(self_, List<optional<Tensor>>(indices_), values_, accumulate);
}

// plumbing done since we don't support List<optional<Tensor>> in codegen
Tensor& index_put__plumbing(Tensor & self, const List<optional<Tensor>> & indices
, const Tensor & values, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return self.index_put_(indices, values, accumulate);
  }
  Tensor self_value, values_value;
  optional<int64_t> self_bdim, values_bdim;
  std::vector<optional<Tensor>> indices_value;
  std::vector<optional<int64_t>> indices_bdims;
  std::tie(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim) =
      unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
  index_put__batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate);
  return self;
}

void _index_put_impl__batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate,
    bool unsafe) {
  if (!self_bdim.has_value()) {
    vmapIncompatibleInplaceError("_index_put_impl_");
  }
  Tensor self_, values_;
  std::vector<optional<Tensor>> indices_;
  std::tie(self_, indices_, values_) = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim);
  at::_index_put_impl_(self_, List<optional<Tensor>>(indices_), values_, accumulate, unsafe);
}

// plumbing done since we don't support List<optional<Tensor>> in codegen
Tensor &_index_put_impl__plumbing(Tensor &self, const List<optional<Tensor>> &indices,
                                  const Tensor &values, bool accumulate, bool unsafe) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return at::_index_put_impl_(self, indices, values, accumulate, unsafe);
  }
  Tensor self_value, values_value;
  optional<int64_t> self_bdim, values_bdim;
  std::vector<optional<Tensor>> indices_value;
  std::vector<optional<int64_t>> indices_bdims;
  std::tie(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim) =
      unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
  _index_put_impl__batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate, unsafe);
  return self;
}

static Tensor maybe_permute_values(
    const Tensor& values,
    ArrayRef<optional<Tensor>> orig_indices,
    ArrayRef<optional<int64_t>> orig_indices_bdims) {
  bool indices_batched = any_has_value(orig_indices_bdims);
  bool advanced_indices_are_adjacent = are_advanced_indices_adjacent(orig_indices);
  auto num_leading_nones = get_num_leading_nones(orig_indices);
  auto max_index_dim = get_max_index_logical_dim(orig_indices, orig_indices_bdims);
  TORCH_INTERNAL_ASSERT(values.dim() >= num_leading_nones + max_index_dim);

  // NB: values has its B dimension at the front
  if (!indices_batched) {
    if (advanced_indices_are_adjacent) {
      // self: Tensor[B, 5, 6, 7, 8]
      // indices: [:, Tensor[2, 2], Tensor[2, 2], :]
      // batched_indices: [:, :, Tensor[2, 2], Tensor[2, 2], :]
      // required values: Tensor[B, 5, 2, 2, 8]
      return values;
    }
    // self: Tensor[B, 5, 6, 7]
    // indices: [Tensor[2, 2], :, Tensor[2, 2]]
    // batched_indices: [:, Tensor[2, 2], :, Tensor[2, 2]]
    // required values: Tensor[2, 2, B, 6]
    return values.movedim(0, max_index_dim);
  }
  if (!advanced_indices_are_adjacent) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [:, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // batched_indices: [arange(B).expand(B, 2, 2), :, Tensor[B, 2, 2], :, Tensor[2, 2]]
    // required values: Tensor[B, 2, 2, 5, 7]
    return values;
  }
  // In other words, in batched_indices, advanced indices are adjacent
  if (num_leading_nones == 0) {
    // self: Tensor[B, 5, 6, 7, 8]
    // indices: [Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // batched_indices: [arange(B).expand(B, 2, 2), Tensor[B, 2, 2], Tensor[2, 2], :, :]
    // required values: Tensor[B, 2, 2, 7, 8]
    return values;
  }
  // This is the tricky case. In indices, advanced indices are adjacent.
  // In batched_indices, advanced indices are no longer adjacent
  //
  // self: Tensor[B, 5, 6, 7, 8, 9]
  // indices: [:, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // batched_indices: [arange(B).expand(B, 2, 3), :, :, Tensor[B, 2, 3], Tensor[2, 3], :]
  // required values: Tensor[B, 2, 3, 5, 6, 9]
  // actual values: Tensor[B, 5, 6, 2, 3, 9]
  //
  // The resolution is to move dims around until we get the right shape.
  // The values is set up as [B, <leading_nones>, <maxIndexDim>, ...]
  // we just have to move the <maxIndexDim> to before the <leading_nones> to produce
  // [B, <maxIndexDim>, <leading_nones>, ...]
  return swap_regions(values, num_leading_nones, max_index_dim);
}

std::tuple<Tensor,optional<int64_t>> index_put_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    ArrayRef<optional<Tensor>> indices,
    ArrayRef<optional<int64_t>> indices_bdims,
    const Tensor& values,
    optional<int64_t> values_bdim,
    bool accumulate) {
  TORCH_INTERNAL_ASSERT(indices.size() == indices_bdims.size());

  // find the batch_size
  int64_t batch_size = 0;
  if (self_bdim || values_bdim) {
    batch_size = get_bdim_size2(self, self_bdim, values, values_bdim);
  } else {
    // one or more of the indices is batched.
    for (size_t i = 0; i < indices.size(); i++) {
      if (indices_bdims[i] && indices[i].has_value()) {
        batch_size = indices[i].value().size(*indices_bdims[i]);
        break;
      }
    }
  }

  Tensor self_, values_;
  std::vector<optional<Tensor>> indices_;
  std::tie(self_, indices_, values_) = index_put_batch_rule_helper(
      self, self_bdim, indices, indices_bdims, values, values_bdim, batch_size);

  // Why do we need to permute values?
  // See NOTE [Advanced indexing (index.Tensor) batch rule] for details,
  // but the gist is that index_put effectively does the following:
  // - result = self_.clone()
  // - result[indices_] = values
  // - return result
  // Now, the problem is, result[indices_] might return a Tensor whose shape is
  // the shape of values, but permuted. This is because the shape of result[indices_]
  // depends on if the original indices "have adjacent advanced indices"
  // and the batched `indices_` might change the "have adjacent advanced indices" property
  values_ = maybe_permute_values(values_, indices, indices_bdims);

  auto result = at::index_put(self_, List<optional<Tensor>>(indices_), values_, accumulate);
  return std::make_tuple(result, 0);
}

// plumbing done since we don't support List<optional<Tensor>> in codegen
Tensor index_put_plumbing(const Tensor & self, const List<optional<Tensor>> & indices,
                          const Tensor & values, bool accumulate) {
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t cur_level = maybe_layer->layerId();
  if (!isBatchedAtLevel(self, cur_level) && !isBatchedAtLevel(indices, cur_level) && !isBatchedAtLevel(values, cur_level)) {
    return self.index_put(indices, values, accumulate);
  }
  Tensor self_value, values_value;
  optional<int64_t> self_bdim, values_bdim;
  std::vector<optional<Tensor>> indices_value;
  std::vector<optional<int64_t>> indices_bdims;
  std::tie(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim) =
      unpackSelfAndIndicesAndValuesAtCurrentLevel(self, indices, values, cur_level);
  auto results = index_put_batch_rule(self_value, self_bdim, indices_value, indices_bdims, values_value, values_bdim, accumulate);
  return makeBatched(std::get<0>(results), std::get<1>(results), cur_level);
}

namespace {

template<typename Func, typename ...Args>
std::tuple<Tensor,optional<int64_t>> scatter_batch_rule(
    Func f,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value, Args... args) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);

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

  auto result = f(self_, physical_dim, index_, value, args...);
  // result should have same shape as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

template <typename Func, typename ...Args>
inline std::tuple<Tensor,optional<int64_t>> scatter_batch_rule(
    Func f,
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim, Args... args) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto src_logical_rank = rankWithoutBatchDim(src, src_bdim);
  auto batch_size = get_bdim_size3(self, self_bdim, index, index_bdim, src, src_bdim);

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

  auto result = f(self_, physical_dim, index_, src_, args...);
  // result should have same shape as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

} // namespace

std::tuple<Tensor,optional<int64_t>> scatter_value_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& value) {
  return scatter_batch_rule(ATEN_FN2(scatter, value),
                            self, self_bdim, dim, index, index_bdim, value);
}

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

std::tuple<Tensor,optional<int64_t>> scatter_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& src, optional<int64_t> src_bdim,
    const c10::string_view reduce) {
  return scatter_batch_rule(ATEN_FN2(scatter, reduce),
                            self, self_bdim, dim, index, index_bdim, src, src_bdim, reduce);
}

std::tuple<Tensor,optional<int64_t>> scatter_value_reduce_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Scalar& src,
    const c10::string_view reduce) {
  return scatter_batch_rule(ATEN_FN2(scatter, value_reduce),
                            self, self_bdim, dim, index, index_bdim, src, reduce);
}

std::tuple<Tensor,optional<int64_t>> gather_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    bool sparse_grad) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto index_logical_rank = rankWithoutBatchDim(index, index_bdim);
  auto batch_size = get_bdim_size2(self, self_bdim, index, index_bdim);

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
  // result should have same rank as index
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
  auto batch_size = get_bdim_size3(grad, grad_bdim, self, self_bdim, index, index_bdim);
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
  // result should has same rank as self
  if (self_logical_rank == 0) {
    result = result.squeeze(-1);
  }
  return std::make_tuple(result, 0);
}

namespace {
Tensor get_expanded_index(const Tensor& index, IntArrayRef self_size, int64_t dim) {
  if (index.dim() == 0) {
    return index.expand(self_size);
  }

  // setup new_index_shape as [BS, 1, ..., idx_size, ..., 1]
  // to reshape index_
  auto idx_size = index.size(0);  // get non-batch size of index tensor
  Tensor index_;
  {
    VmapDimVector new_index_shape(self_size.size(), 1);
    new_index_shape[dim] = idx_size;
    index_ = index.view(new_index_shape);
  }
  // Now apply expand to index_
  {
    VmapDimVector new_index_shape = {self_size.begin(), self_size.end()};
    new_index_shape[dim] = idx_size;
    index_ = index_.expand(new_index_shape);
  }
  return index_;
}
}

Tensor index_select_decomp(const Tensor &self, int64_t dim, const Tensor &index)
{
  Tensor index_ = index;
  if (self.dim() > index.dim()) {
    index_ = get_expanded_index(index, self.sizes(), dim);
  }

  auto result = at::gather(self, dim, index_);

  // output of gather has same dimension as `index` while
  // output of index_select has same dimension as self
  // Eg. t = torch.tensor(1)
  //     idx = torch.tensor([0])
  //     torch.index_select(t, 0, idx) # 0-D
  //     torch.gather(t, 0, idx) # 1-D
  if (self.dim() == 0 && result.dim() != 0) {
    result = result.squeeze(-1);
  }

  return result;
}

Tensor index_copy_decomp(
    const Tensor &self, int64_t dim,
    const Tensor &index, const Tensor &source)
{
  Tensor index_ = index;
  if (self.dim() > index.dim()) {
    index_ = get_expanded_index(index, self.sizes(), dim);
  }

  return at::scatter(self, dim, index_, source);  ;
}

// Note [Fix vmap slice_scatter]
// registers a decomposition for `slice_scatter` that calls into `slice.src`
// *_scatter operators have some special semantics though, that we can't easily
// through a decomposition: slice_scatter's output needs to have the same
// size, size, strides and storage_offset as the input.
Tensor slice_scatter_decomp(const Tensor &self, const Tensor &src,
                            int64_t dim, c10::optional<int64_t> start,
                            c10::optional<int64_t> end, int64_t step)
{
  auto idx = at::arange(start.value_or(0), end.value_or(self.size(dim)), step, self.options().dtype(kLong));
  idx = get_expanded_index(idx, self.sizes(), dim);
  return at::scatter(self, dim, idx, src);
}

Tensor select_scatter_decomp(
    const Tensor &self, const Tensor &source,
    int64_t dim, int64_t index)
{
  // supports negative index
  index = maybe_wrap_dim(index, self.size(dim));
  auto index_ = at::scalar_tensor(index, self.options().dtype(kLong));

  return at::scatter(self, dim, index_.expand_as(self), source.unsqueeze(dim).expand_as(self));
}

std::tuple<Tensor, optional<int64_t>> diagonal_scatter_batch_rule(
    const Tensor &self, c10::optional<int64_t> self_bdim,
    const Tensor &src, c10::optional<int64_t> src_bdim,
    int64_t offset, int64_t dim1, int64_t dim2)
{
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto src_ = moveBatchDimToFront(src, src_bdim);

  auto batch_size = get_bdim_size2(self, self_bdim, src, src_bdim);

  self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
  src_ = ensure_has_bdim(src_, src_bdim.has_value(), batch_size);

  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  dim1 = maybe_wrap_dim(dim1, self_logical_rank) + 1;
  dim2 = maybe_wrap_dim(dim2, self_logical_rank) + 1;

  return std::make_tuple(at::diagonal_scatter(self_, src_, offset, dim1, dim2), 0);
}

std::tuple<Tensor,optional<int64_t>> index_add_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    int64_t dim,
    const Tensor& index, optional<int64_t> index_bdim,
    const Tensor& other, optional<int64_t> other_bdim,
    const Scalar& alpha) {
  if (!index_bdim) {
    // Handle scalar tensors... self, other can be scalar tensors
    const auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
    const auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
    auto self_ = moveBatchDimToFront(self, self_bdim);
    if (self_logical_rank == 0) {
      self_ = self_.unsqueeze(-1);
    }
    auto other_ = moveBatchDimToFront(other, other_bdim);
    if (other_logical_rank == 0) {
      other_ = other_.unsqueeze(-1);
    }
    dim = maybe_wrap_dim(dim, self_logical_rank);

    const auto batch_size = get_bdim_size2(self, self_bdim, other, other_bdim);
    self_ = ensure_has_bdim(self_, self_bdim.has_value(), batch_size);
    other_ = ensure_has_bdim(other_, other_bdim.has_value(), batch_size);

    auto result = self_.index_add(dim + 1, index, other_, alpha);
    if (self_logical_rank == 0) {
      result = result.squeeze(-1);
    }
    return std::make_tuple(result, 0);
  }

  // Index is batched. For-loop and stack is the best thing I can come up with
  // right now. We really want generalized index_add kernel in PyTorch
  auto batch_size = get_bdim_size3(self, self_bdim, other, other_bdim, index, index_bdim);
  std::vector<Tensor> results;
  results.reserve(batch_size);
  for (const auto i : c10::irange(0, batch_size)) {
    const auto& self_slice = self_bdim.has_value() ?
      self.select(*self_bdim, i) : self;
    const auto& other_slice = other_bdim.has_value() ?
      other.select(*other_bdim, i) : other;
    const auto& index_slice = index_bdim.has_value() ?
      index.select(*index_bdim, i) : index;
    results.push_back(at::index_add(self_slice, dim, index_slice, other_slice, alpha));
  }
  return std::make_tuple(at::stack(results), 0);
}

static std::tuple<Tensor,Tensor> binary_pointwise_align(
    const Tensor & self,
    optional<int64_t> self_bdim,
    const Tensor & mask,
    optional<int64_t> mask_bdim) {
  // compute max logical rank
  auto tensor_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(mask, mask_bdim);
  auto max_logical_rank = std::max(tensor_logical_rank, other_logical_rank);

  auto tensor_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(mask, mask_bdim);

  // If the dimensions aren't aligned, we need to line them up.
  // Tensor[B, 3] + Tensor[2, 5, 3] -> Tensor[B, 1, 1, 3] + Tensor[2, 5, 3]
  // Note that only tensors that have a batch dim need to be modified.
  // Tensor[B, 2, 3, 5] + Tensor[5] -> no changes needed
  tensor_ = maybePadToLogicalRank(tensor_, self_bdim, max_logical_rank);
  other_ = maybePadToLogicalRank(other_, mask_bdim, max_logical_rank);

  return std::make_tuple(tensor_, other_);
}

std::tuple<Tensor,optional<int64_t>> masked_fill_scalar_batch_rule(
    const Tensor & self,
    optional<int64_t> self_bdim,
    const Tensor & mask,
    optional<int64_t> mask_bdim,
    const Scalar& source) {
  auto tensors = binary_pointwise_align(self, self_bdim, mask, mask_bdim);
  auto result = at::masked_fill(std::get<0>(tensors), std::get<1>(tensors), source);
  return std::make_tuple(result, 0);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  m.impl("index.Tensor", index_plumbing);
  m.impl("index_put_", index_put__plumbing);
  m.impl("index_put", index_put_plumbing);
  m.impl("_index_put_impl_", _index_put_impl__plumbing);
  m.impl("slice_scatter", slice_scatter_decomp);
  m.impl("select_scatter", select_scatter_decomp);
  m.impl("index_copy", index_copy_decomp);
  m.impl("index_select", index_select_decomp);
  VMAP_SUPPORT2(masked_fill, Scalar, masked_fill_scalar_batch_rule);
  VMAP_SUPPORT(index_add, index_add_batch_rule);
  VMAP_SUPPORT(diagonal_scatter, diagonal_scatter_batch_rule);
  VMAP_SUPPORT(gather, gather_batch_rule);
  VMAP_SUPPORT(gather_backward, gather_backward_batch_rule);
  VMAP_SUPPORT2(scatter, value, scatter_value_batch_rule);
  VMAP_SUPPORT2(scatter, src, scatter_src_batch_rule);
  VMAP_SUPPORT(scatter_add, scatter_add_batch_rule);
  VMAP_SUPPORT2(scatter, reduce, scatter_reduce_batch_rule);
  VMAP_SUPPORT2(scatter, value_reduce, scatter_value_reduce_batch_rule);
}

}}
