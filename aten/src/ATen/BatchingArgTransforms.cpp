#include <ATen/BatchingArgTransforms.h>
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
  std::vector<int64_t> permutation(sizes.size(), 0);
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

PhysicalView MultiBatchArgTransform::logicalToPhysical(const Tensor& logical_tensor) {
  auto* batched = maybeGetBatched(logical_tensor);
  TORCH_INTERNAL_ASSERT(
      batched,
      "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  return { permuteBatchDimsToFront(batched), createLevelsBitset(batched->bdims()) };
}

std::vector<PhysicalView>
MultiBatchArgTransform::logicalToPhysical(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

int64_t PhysicalView::numBatchDims() {
  return levels_.count();
}

int64_t PhysicalView::numLogicalDims() {
  return /*physical*/tensor_.dim() - numBatchDims();
}

std::vector<int64_t> PhysicalView::getPhysicalDims(IntArrayRef logical_dims) {
  auto logical_ndim = numLogicalDims();
  return fmap(
      logical_dims,
      [&](int64_t dim) -> int64_t {
          return maybe_wrap_dim(dim, logical_ndim) + numBatchDims();
      });
}

int64_t PhysicalView::getPhysicalDim(int64_t logical_dim) {
  auto logical_ndim = numLogicalDims();
  return maybe_wrap_dim(logical_dim, logical_ndim) + numBatchDims();
}


} // namespace at
