#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace at::native {

TORCH_CUDA_CPP_API bool _scaled_grouped_mm_allowed_device();

} // namespace at::native
