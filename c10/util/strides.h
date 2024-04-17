#pragma once
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>

namespace c10 {

// Computes the contiguous strides of a tensor, given its sizes.
static inline DimVector contiguous_strides(const IntArrayRef sizes) {
  using Int = IntArrayRef::value_type;
  const Int dims = static_cast<Int>(sizes.size());

  // With this initialisation we get the case dim == 0 or 1 right
  DimVector strides(dims, 1);

  for (auto i = dims - 2; i >= 0; --i) {
    // Strides can't be 0 even if sizes are 0.
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], Int{1});
  }

  return strides;
}

} // namespace c10
