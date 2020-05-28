#include <ATen/BatchingUtils.h>
#include <ATen/ATen.h>

namespace at {

static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    if (bdims[idx].dim() != idx) {
      return false;
    }
  }
  return true;
}

std::bitset<kVmapNumLevels> createLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapNumLevels> result;
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

std::vector<int64_t> transformIntVector(IntArrayRef arr, std::function<int64_t(int64_t)> func) {
  std::vector<int64_t> result;
  result.reserve(arr.size());
  for (int64_t elt : arr) {
    result.push_back(func(elt));
  }
  return result;
}

std::pair<Tensor,std::bitset<kVmapNumLevels>>
materializeBatchDimsAtFront(const Tensor& tensor) {
  auto* batched = maybeGetBatched(tensor);
  if (!batched) {
    return std::make_pair(tensor, 0);
  }
  auto bdims = batched->bdims();
  const Tensor& tensor_ = batched->value();
  auto levels = createLevelsBitset(bdims);
  if (areBdimsAtFrontInOrder(bdims)) {
    return std::make_pair(tensor_, levels);
  }
  const auto sizes = tensor_.sizes();
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
  return std::make_pair(tensor_.permute(permutation), levels);
}

}
