#pragma once

#include <ATen/Half.h>
#include <ATen/cuda/CUDAHalf.cuh>

namespace at {
template<typename T>
struct to_cuda_type { using type = T; };

template<> struct to_cuda_type<Half> { using type = half; };
} // namespace at
