#include <ATen/BatchingArgTransforms.h>
#include <ATen/ATen.h>

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
  return { permuteBatchDimsToFront(batched) };
}

std::vector<PhysicalView>
MultiBatchArgTransform::logicalToPhysical(TensorList logical_tensors) {
  TORCH_INTERNAL_ASSERT(false, "NYI");
}

} // namespace at
