#pragma once
#include <cstdint>
#include <ATen/core/TensorBase.h>

namespace at {
namespace native {

// Stable-sort self into values, and set indices to the
// inverse-permutation from values back to self.
// Output tensors must be pre-allocated and contiguous.
void launch_stable_sort_kernel(const TensorBase &self, int64_t dim, bool descending,
                               const TensorBase &values, const TensorBase &indices);

inline bool should_use_small_sort(int64_t nsort) {
  return nsort <= 4096;
}

inline bool should_use_small_sort(const TensorBase &self, int64_t dim) {
  return should_use_small_sort(self.size(dim));
}

void sortKeyValueInplace(
    const TensorBase &key, const TensorBase &value, int dim,
    bool descending, bool stable=false);

}}  // namespace at::native
