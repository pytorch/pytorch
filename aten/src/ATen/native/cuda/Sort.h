#pragma once
#include <cstdint>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cuda/SortStable.h>

namespace at {
namespace native {

inline bool should_use_small_sort(const TensorBase &self, int64_t dim) {
  return self.size(dim) <= 4096;
}

void sortKeyValueInplace(
    const TensorBase &key, const TensorBase &value, int dim,
    bool descending, bool stable=false);

}}  // namespace at::native
