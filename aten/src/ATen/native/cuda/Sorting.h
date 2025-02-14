#pragma once
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

void launch_kthvalue_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t k);
void launch_median_kernel(
    const TensorBase &vals, const TensorBase &inds,
    const TensorBase &in, int64_t dim, bool ignore_nan);

}  // namespace at::native
