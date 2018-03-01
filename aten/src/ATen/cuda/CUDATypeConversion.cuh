#pragma once

#include <ATen/Half.h>
#include <ATen/cuda/CUDAHalf.cuh>

// Type traits to convert types to CUDA-specific types. Used primarily to
// convert at::Half to CUDA's half type. This makes the conversion explicit.

namespace at {
template<typename T>
struct to_cuda_type { using type = T; };

template<> struct to_cuda_type<Half> { using type = half; };
} // namespace at
