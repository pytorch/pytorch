// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/DynamicLayer.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

namespace at {
namespace functorch {

// Checks if the batch dims in `bdims` appear at the front of the tensor.
static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (const auto idx: c10::irange(0, bdims.size())) {
    if (bdims[idx].dim() != (int64_t)idx) {
      return false;
    }
  }
  return true;
}

// Takes a BatchedTensorImpl, permutes all of the batch dims to the front,
// and then returns a physical version of the Tensor.
static Tensor permuteBatchDimsToFront(BatchedTensorImpl* batched) {
  auto bdims = batched->bdims();
  const Tensor& physical_tensor = batched->value();
  if (areBdimsAtFrontInOrder(bdims)) {
    return physical_tensor;
  }
  const auto sizes = physical_tensor.sizes();
  VmapDimVector permutation(sizes.size(), 0);
  permutation.reserve(sizes.size());
  const auto is_bdim = createBatchDimBitset(bdims);
  int64_t idx = 0;
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.dim();
  }
  for (const auto ptr : c10::irange(0, sizes.size())) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = ptr;
  }
  return physical_tensor.permute(permutation);
}

VmapPhysicalView MultiBatchVmapTransform::logicalToPhysical(const Tensor& logical_tensor) {
  auto* batched = maybeGetBatchedImpl(logical_tensor);
  TORCH_INTERNAL_ASSERT(
      batched,
      "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  return { permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims()) };
}

int64_t VmapPhysicalView::numBatchDims() const {
  return levels_.count();
}

int64_t VmapPhysicalView::numLogicalDims() const {
  return /*physical*/tensor_.dim() - numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalDims(IntArrayRef logical_dims) const {
  auto logical_ndim = numLogicalDims();
  // NB: fmap doesn't have a SmallVector variant, so we don't use it here.
  VmapDimVector result;
  result.reserve(logical_ndim);
  for (auto dim : logical_dims) {
    result.push_back(maybe_wrap_dim(dim, logical_ndim) + numBatchDims());
  }
  return result;
}

int64_t VmapPhysicalView::getPhysicalDim(int64_t logical_dim) const {
  auto logical_ndim = numLogicalDims();
  return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalShape(IntArrayRef logical_shape) const {
  VmapDimVector result;
  result.reserve(logical_shape.size() + numBatchDims());
  auto tensor_sizes = tensor_.sizes();
  result.insert(result.end(), tensor_sizes.begin(), tensor_sizes.begin() + numBatchDims());
  result.insert(result.end(), logical_shape.begin(), logical_shape.end());
  return result;
}

static BatchDims computeFrontBatchDimsFromLevels(std::bitset<kVmapNumLevels> levels_bitset) {
  BatchDims bdims;
  int64_t dim = 0;
  for (int64_t level = 0; level < kVmapNumLevels; level++) {
    if (!levels_bitset[level]) {
      continue;
    }
    bdims.emplace_back(level, dim++);
  }
  return bdims;
}

static Tensor moveDimToFrontAndExpand(Tensor tensor, optional<int64_t> dim, int64_t size) {
  if (dim) {
    tensor = tensor.movedim(*dim, 0);
  } else {
    tensor = tensor.unsqueeze(0);
    auto expanded_sizes = tensor.sizes().vec();
    expanded_sizes[0] = size;
    tensor = tensor.expand(expanded_sizes);
  }
  return tensor;
}

// The algorithm is as follows:
// 1. Figure out what all of the collective levels in `logical_tensors` is.
// 2. Move all batch dims to the front of the tensors and add extra dims
//    of size 1. At this point, every tensor will have a dimension for
//    each of the collective levels.
// 3. Compute the batch_sizes.
// 4. Expand each physical tensor so that they have output batch size equal
//    to `batch_sizes`
VmapPhysicalViewVec
MultiBatchVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  auto cur_level = maybeCurrentDynamicLayer().value().layerId();
  auto bdim_size = -1;

  // Figure out the batch size first
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(batched->bdims().size() == 1);
    if (batched->bdims().back().level() != cur_level) {
      continue;
    }
    bdim_size = batched->value().size(batched->bdims().back().dim());
  }
  TORCH_INTERNAL_ASSERT(bdim_size != -1);

  std::bitset<kVmapNumLevels> levels;
  levels[cur_level] = 1;

  VmapPhysicalViewVec result;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched || (batched->bdims().back().level() != cur_level)) {
      // Unsqueeze dim 0, expand it to the correct shape
      c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
      auto value = moveDimToFrontAndExpand(logical_tensor, {}, bdim_size);
      result.emplace_back(std::move(value), levels);
      continue;
    }
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto physical = batched->value();
    auto value = moveDimToFrontAndExpand(physical, batched->bdims().back().dim(), bdim_size);
    result.emplace_back(std::move(value), levels);
  }

  return result;
}

static Tensor moveDimToFrontAndUnsqueeze(Tensor tensor, optional<int64_t> dim, int64_t example_ndim) {
  if (dim) {
    tensor = tensor.movedim(*dim, 0);
  } else {
    tensor = tensor.unsqueeze(0);
  }
  auto ndim = tensor.dim() - 1;
  for (int64_t i = 0; i < example_ndim - ndim; i++) {
    tensor = tensor.unsqueeze(1);
  }
  return tensor;
}

VmapPhysicalViewVec BroadcastingVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  auto cur_level = maybeCurrentDynamicLayer().value().layerId();
  auto bdim_size = -1;

  // Figure out the batch size first
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched || (batched->bdims().back().level() != cur_level)) {
      continue;
    }
    bdim_size = batched->value().size(batched->bdims().back().dim());
  }
  TORCH_INTERNAL_ASSERT(bdim_size != -1);

  std::bitset<kVmapNumLevels> levels;
  levels[cur_level] = 1;

  // figure out the example ndim
  int64_t max_example_dim = -1;
  for (const auto& logical_tensor : logical_tensors) {
    max_example_dim = std::max(logical_tensor.dim(), max_example_dim);
  }

  VmapPhysicalViewVec result;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched || (batched->bdims().back().level() != cur_level)) {
      // Unsqueeze dim 0, expand it to the correct shape
      c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
      auto value = moveDimToFrontAndUnsqueeze(logical_tensor, {}, max_example_dim);
      result.emplace_back(std::move(value), levels);
      continue;
    }
    c10::impl::ExcludeDispatchKeyGuard guard(kBatchedKey);
    auto physical = batched->value();
    auto value = moveDimToFrontAndUnsqueeze(physical, batched->bdims().back().dim(), max_example_dim);
    result.emplace_back(std::move(value), levels);
  }

  return result;
}

VmapPhysicalToLogicalMap VmapPhysicalView::getPhysicalToLogicalMap() const {
  return VmapPhysicalToLogicalMap(levels_);
}

Tensor VmapPhysicalToLogicalMap::apply(const Tensor& physical_tensor) const {
  return makeBatched(physical_tensor, computeFrontBatchDimsFromLevels(levels_));
}

void VmapPhysicalToLogicalMap::applyInplace(std::vector<Tensor>& physical_tensors) const {
  for (const auto idx : c10::irange(0, physical_tensors.size())) {
    physical_tensors[idx] = apply(physical_tensors[idx]);
  }
}

}
} // namespace at
