#include <ATen/VmapTransforms.h>
#include <ATen/ATen.h>

namespace at {

// Creates a bitset for all of the levels present in `bdims`.
std::bitset<kVmapNumLevels> createLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapNumLevels> result;
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

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
  auto* batched = maybeGetBatched(logical_tensor);
  TORCH_INTERNAL_ASSERT(
      batched,
      "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  return { permuteBatchDimsToFront(batched), createLevelsBitset(batched->bdims()) };
}

std::vector<VmapPhysicalView>
MultiBatchVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

int64_t VmapPhysicalView::numBatchDims() {
  return levels_.count();
}

int64_t VmapPhysicalView::numLogicalDims() {
  return /*physical*/tensor_.dim() - numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalDims(IntArrayRef logical_dims) {
  auto logical_ndim = numLogicalDims();
  // NB: fmap doesn't have a SmallVector variant, so we don't use it here.
  VmapDimVector result;
  result.reserve(logical_ndim);
  for (auto dim : logical_dims) {
    result.push_back(maybe_wrap_dim(dim, logical_ndim) + numBatchDims());
  }
  return result;
}

int64_t VmapPhysicalView::getPhysicalDim(int64_t logical_dim) {
  auto logical_ndim = numLogicalDims();
  return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalShape(IntArrayRef logical_shape) {
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

Tensor VmapPhysicalView::newLogicalFromPhysical(const Tensor& physical) {
  return makeBatched(physical, computeFrontBatchDimsFromLevels(levels_));
}

// Given a Tensor or a BatchedTensor, returns the underlying physical tensor
// with all vmapped dimensions permuted to the front, if they exist, and a
// bitset of vmap levels that were present in the tensor.
static std::pair<Tensor,std::bitset<kVmapNumLevels>>
getPhysicalTensorAndLevels(const Tensor& self) {
  auto* batched = maybeGetBatched(self);
  if (batched) {
    return {permuteBatchDimsToFront(batched), createLevelsBitset(batched->bdims())};
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

  std::vector<int64_t> aligned_sizes(requested_levels.count() + requested_example_dim, 1);

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

static std::pair<std::bitset<kVmapNumLevels>,int64_t>
getLevelsAndLargestLogicalDim(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(logical_tensors.size() > 0);
  std::bitset<kVmapNumLevels> levels;
  int64_t largest_logical_dim = -1;
  for (const auto& tensor : logical_tensors) {
    auto* batched = maybeGetBatched(tensor);
    if (batched) {
      levels = levels | createLevelsBitset(batched->bdims());
    }
    auto tensor_logical_dim = /*logical dim*/tensor.dim();
    if (tensor_logical_dim > largest_logical_dim) {
      largest_logical_dim = tensor_logical_dim;
    }
  }
  return { levels, largest_logical_dim };
}

VmapPhysicalViewVec BroadcastingVmapTransform::logicalToPhysical(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(
      logical_tensors.size() == 2,
      "This function has only been tested for two tensors. Please add more tests ",
      "before removing this check ");

  VmapPhysicalViewVec result;

  std::bitset<kVmapNumLevels> levels;
  int64_t largest_logical_dim;
  std::tie(levels, largest_logical_dim) = getLevelsAndLargestLogicalDim(logical_tensors);

  for (const auto& tensor : logical_tensors) {
    // NB: It's possible that we didn't actually need to align `tensor`.
    // For example, when adding two tensors of size (B, 2), and (3, 2), where
    // the first Tensor is a BatchedTensor with batch dim B and the second is
    // a regular Tensor, we will return views of size (B, 1, 2) and (1, 3, 2).
    // However, the view on the second tensor is unnecessary: broadcasting
    // semantics allow for the addition of two tensors of size (B, 1, 2) and (3, 2)!
    //
    // If this unnecessary view is a problem, consider optimizing it away in
    // the future. This may involve creating a new type of VmapPhysicalView
    auto aligned = alignBatchDimsAtFront(tensor, levels, largest_logical_dim) ;
    result.emplace_back(std::move(aligned), levels);
  }
  return result;
}

} // namespace at
