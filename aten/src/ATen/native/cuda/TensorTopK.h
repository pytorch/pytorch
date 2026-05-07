#pragma once
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {
void launch_gather_topk_kernel(
    const TensorBase& self,
    int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices);
}
