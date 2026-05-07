#pragma once
#include <ATen/core/TensorBase.h>
#include <cstdint>

namespace at::native {

// Stable-sort self into values, and set indices to the
// inverse-permutation from values back to self.
// Output tensors must be pre-allocated and contiguous.
void launch_stable_sort_kernel(
    const TensorBase& self,
    int64_t dim,
    bool descending,
    const TensorBase& values,
    const TensorBase& indices);

} // namespace at::native
