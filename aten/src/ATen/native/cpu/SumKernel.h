#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

// Sums |x|^p over the reduced dimensions (for p == 0, the number of nonzero
// elements) with the cascade summation sum() uses, so the result does not stop
// growing once the running total outgrows the accumulator's mantissa. p must be
// 0, 1 or 2, and the input a real floating point type. The output dtype must be
// the input dtype for float and double, and float for half and bfloat16 unless
// the output has a single element. With is_norm, the output is the p-norm, so
// the square root of the sum for p == 2.
void cascade_powsum_kernel(TensorIterator& iter, double p, bool is_norm);

} // namespace CPU_CAPABILITY
} // namespace at::native
