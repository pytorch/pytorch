#include <ATen/VmapTransforms.h>
#include <ATen/ATen.h>
#include <ATen/DynamicLayer.h>

namespace at {

// Checks if the batch dims in `bdims` appear at the front of the tensor.
static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    if (bdims[idx].dim() != idx) {
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
  for (int64_t ptr = 0; idx < sizes.size(); ptr++) {
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

// Given a Tensor or a BatchedTensor, returns the underlying physical tensor
// with all vmapped dimensions permuted to the front, if they exist, and a
// bitset of vmap levels that were present in the tensor.
static std::pair<Tensor,std::bitset<kVmapNumLevels>>
getPhysicalTensorAndLevels(const Tensor& self) {
  auto* batched = maybeGetBatchedImpl(self);
  if (batched) {
    return {permuteBatchDimsToFront(batched), createVmapLevelsBitset(batched->bdims())};
  }
  return {self, 0};
}

// Given a Tensor or a BatchedTensor, creates a physical view of the tensor
// such that it has a batch dimension for each level in `requested_levels`
// and `requested_example_dim` number of non-batch-dimensions.
//
// This function is useful in preparing physical views on tensors that can
// then be passed into broadcasting operations. For example, when adding
// two BatchedTensors of sizes [B0, 3] and [B0, B1, 2, 3], where the Bi are the
// batch dimensions, we must align the batch dimensions and non-batch-dimensions
// (henceforth referred to as the "example" dimensions) separately to produce
// tensors of size [B0, 1, 1, 3] and [B0, B1, 2, 3] so that they can be added.
//
// Here's a direct example of using alignBatchDimsAtFront on the above two tensors.
//
// 1) alignBatchDimsAtFront([B0, 3], requested_levels={0, 1}, requested_example_dim=2)
// returns a physical view of size [B0, 1, 1, 3] by adding an extra dimension for
// level 1 and another extra dimension to pad the example dimensions to 2.
//
// 2) alignBatchDimsAtFront([B0, B1, 2, 3], requested_levels={0, 1}, requested_example_dim=2)
// returns a physical view of size [B0, B1, 2, 3]
static Tensor alignBatchDimsAtFront(
    const Tensor& self,
    std::bitset<kVmapNumLevels> requested_levels,
    int64_t requested_example_dim) {
  Tensor physical_tensor;
  std::bitset<kVmapNumLevels> tensor_levels;
  std::tie(physical_tensor, tensor_levels) = getPhysicalTensorAndLevels(self);

  TORCH_INTERNAL_ASSERT(
    (tensor_levels | requested_levels) == requested_levels,
    "`requested_levels` must be a superset of `self`'s levels");

  auto physical_sizes = physical_tensor.sizes();

  auto tensor_example_dim = physical_sizes.size() - /*num_batch_dims*/tensor_levels.count();
  TORCH_INTERNAL_ASSERT(tensor_example_dim <= requested_example_dim);

  if (tensor_levels == requested_levels && tensor_example_dim == requested_example_dim) {
    // Optimization: no need to do another view if the physical tensor is
    // already the correct shape
    return physical_tensor;
  }

  VmapDimVector aligned_sizes(requested_levels.count() + requested_example_dim, 1);

  // align the example dims (non-bdims dims) first
  // aligned_sizes[-tensor_example_dim:] = tensor_sizes[-tensor_example_dim:]
  std::copy(
      physical_sizes.rbegin(),
      physical_sizes.rbegin() + tensor_example_dim,
      aligned_sizes.rbegin());

  // align the bdims
  int64_t level = 0;
  int64_t tensor_dim = 0;
  for (int64_t bdim = 0; bdim < requested_levels.count(); bdim++) {
    // Determine the level of the bdim
    while (!requested_levels[level]) level++;
    if (tensor_levels[level]) {
      aligned_sizes[bdim] = physical_sizes[tensor_dim++];
    }
    level++;
  }
  return physical_tensor.view(aligned_sizes);
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
  // Figure out the highest vmap level
  int64_t max_vmap_level = -1;
  int64_t batch_size = -1;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (!batched) {
      continue;
    }
    int64_t candidate = batched->bdims().front().level();
    if (candidate < max_vmap_level) {
      continue;
    }
    max_vmap_level = candidate;
    auto bdim = batched->bdims().front().dim();
    batch_size = batched->value().size(bdim);
  }
  TORCH_INTERNAL_ASSERT(max_vmap_level > 0);
  TORCH_INTERNAL_ASSERT(batch_size >= 0);

  std::bitset<kVmapNumLevels> levels;
  levels[max_vmap_level] = 1;

  // unsqueeze extra dim from all tensors and expand it
  VmapPhysicalViewVec result;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(logical_tensor);
    if (batched && batched->bdims().front().level() == max_vmap_level) {
      Tensor tensor = alignBatchDimsAtFront(logical_tensor, levels, logical_tensor.dim());
      result.emplace_back(std::move(tensor), levels);
      continue;
    }
    auto tensor = logical_tensor.unsqueeze(0);
    std::vector<int64_t> expanded_shape = tensor.sizes().vec();
    expanded_shape[0] = batch_size;
    tensor = tensor.expand(expanded_shape);
    result.emplace_back(std::move(tensor), levels);
  }
  return result;
}

static std::pair<std::bitset<kVmapNumLevels>,int64_t>
getLevelsAndLargestLogicalDim(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(logical_tensors.size() > 0);
  std::bitset<kVmapNumLevels> levels;
  int64_t largest_logical_dim = -1;
  for (const auto& tensor : logical_tensors) {
    auto* batched = maybeGetBatchedImpl(tensor);
    if (batched) {
      levels = levels | createVmapLevelsBitset(batched->bdims());
    }
    auto tensor_logical_dim = /*logical dim*/tensor.dim();
    if (tensor_logical_dim > largest_logical_dim) {
      largest_logical_dim = tensor_logical_dim;
    }
  }
  return { levels, largest_logical_dim };
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
      c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Batched);
      auto value = moveDimToFrontAndUnsqueeze(logical_tensor, {}, max_example_dim);
      result.emplace_back(std::move(value), levels);
      continue;
    }
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Batched);
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
  for (int64_t idx = 0; idx < physical_tensors.size(); ++idx) {
    physical_tensors[idx] = apply(physical_tensors[idx]);
  }
}

} // namespace at
