#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace at::native {

TORCH_CUDA_CPP_API bool _scaled_mm_allowed_device(
    bool sm90_only = false,
    bool sm100_only = false);

} // namespace at::native
