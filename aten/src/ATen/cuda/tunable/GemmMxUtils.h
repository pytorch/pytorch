#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <string>

namespace at::cuda::tunable {

#ifdef USE_ROCM
// Forward declaration of the function from ATen/native/cuda/Blas.cpp
bool _is_gfx950_supported();
#endif

// Helper function to validate MX format requirements
static bool ValidateMXFormatRequirements(int64_t m, int64_t n, int64_t k) {
  constexpr int32_t required_block_size = 32;
  return (m % required_block_size == 0) && 
         (n % required_block_size == 0) && 
         (k % required_block_size == 0);
}

} // namespace at::cuda::tunable 