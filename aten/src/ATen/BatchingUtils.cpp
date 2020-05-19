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

// NB: There is an invariant that BatchDims are stored in increasing
// `level` order.
BatchDims moveBatchDimsToFront(BatchDimsRef bdims) {
  BatchDims result;
  result.reserve(bdims.size());
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    result.emplace_back(bdims[idx].level(), idx);
  }
  return result;
}

Tensor moveBatchDimsToFront(const Tensor& self, BatchDimsRef bdims) {
  if (areBdimsAtFrontInOrder(bdims)) {
    return self;
  }
  const auto self_sizes = self.sizes();
  std::vector<int64_t> permutation(self_sizes.size(), 0);
  permutation.reserve(self_sizes.size());
  const auto is_bdim = createBatchDimBitset(bdims);
  int64_t idx = 0;
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.dim();
  }
  for (int64_t ptr = 0; idx < self_sizes.size(); ptr++) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = ptr;
  }
  return self.permute(permutation);
}

std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self) {
  const auto* batched = maybeGetBatched(self);
  if (batched) {
    return { batched->value(), batched->bdims() };
  }
  return { self, {} };
}

}
