#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <string>

namespace at::cuda::tunable {

#ifdef USE_ROCM
static bool IsGfx950Device() {
  // Single static check - only evaluated once
  static bool is_gfx950 = []() {
    auto device = at::cuda::current_device();
    hipDeviceProp_t* prop = at::cuda::getDeviceProperties(device);
    return (std::string(prop->gcnArchName) == "gfx950");
  }();
  return is_gfx950;
}
#endif

// Helper function to validate MX format requirements
static bool ValidateMXFormatRequirements(int64_t m, int64_t n, int64_t k) {
  constexpr int32_t required_block_size = 32;
  return (m % required_block_size == 0) && 
         (n % required_block_size == 0) && 
         (k % required_block_size == 0);
}

} // namespace at::cuda::tunable 