#pragma once

#include <c10/core/Allocator.h>

namespace at { namespace cuda {

TORCH_CUDA_API at::Allocator* getPinnedMemoryAllocator();

}} // namespace at::cuda
