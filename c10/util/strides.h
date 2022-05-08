#pragma once
#include <c10/util/ArrayRef.h>

namespace c10 {

// Computes the contiguous strides of a tensor, given its sizes.
static inline std::vector<typename IntArrayRef::value_type> contiguous_strides(
    const IntArrayRef sizes) {
  using Int = IntArrayRef::value_type;
  const Int dims = static_cast<Int>(sizes.size());

  std::vector<Int> strides;

  if (dims > 0) {
    strides.assign(dims, 0);
    // Start by populating the last dimension: its strides is always 1.
    strides[dims - 1] = 1;
    for (auto i = dims - 2; i >= 0; --i) {
      // Strides can't be 0 even if sizes are 0.
      strides[i] = strides[i + 1] * std::max(sizes[i + 1], Int{1});
    }
  }

  return strides;
}

} // namespace c10
