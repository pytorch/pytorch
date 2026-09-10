#pragma once
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

void launch_fused_mode_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t slice_size, int64_t slices);

void launch_apply_mode_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t ndim);

}  // namespace at::native
