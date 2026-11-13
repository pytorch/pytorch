#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native {

// Whether cascade_powsum_kernel handles this p. _powsum's p == 0 counts zeros
// too (0^0 == 1), so only the norm takes p == 0.
inline bool cascade_powsum_supports(double p, bool is_norm) {
  return p == 1.0 || p == 2.0 || (is_norm && p == 0.0);
}

inline namespace CPU_CAPABILITY {

// Sums |x|^p over the reduced dimensions (for p == 0, the number of nonzero
// elements) with the cascade summation sum() uses, so small terms after a large
// one are kept (#65815). As in sum(), when the reduced dims cannot be merged
// into one (e.g. dim=(0, 2)), the partial sums of the rows are still added to
// the output one after another, so half and bfloat16 need a float output unless
// it has a single element. float and double need the input dtype. With is_norm,
// the output is the p-norm, so the square root of the sum for p == 2. Returns
// false, without touching the output, for other p, input or output dtypes.
bool cascade_powsum_kernel(TensorIterator& iter, double p, bool is_norm);

} // namespace CPU_CAPABILITY
} // namespace at::native
