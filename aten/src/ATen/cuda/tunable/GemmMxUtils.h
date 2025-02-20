#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <unordered_map>
#include <string>

namespace at::cuda::tunable {

// Helper function to cache device properties check
static bool IsGfx950Device() {
  static std::mutex mutex;
  static std::unordered_map<int, bool> device_check_cache;
  
  auto device = at::cuda::current_device();
  
  std::lock_guard<std::mutex> guard(mutex);
  auto it = device_check_cache.find(device);
  if (it != device_check_cache.end()) {
    return it->second;
  }
  
  hipDeviceProp_t* prop = at::cuda::getDeviceProperties(device);
  bool is_gfx950 = (std::string(prop->gcnArchName) == "gfx950");
  device_check_cache[device] = is_gfx950;
  return is_gfx950;
}

// Helper function to validate MX format requirements
static bool ValidateMXFormatRequirements(int64_t m, int64_t n, int64_t k) {
  constexpr int32_t required_block_size = 32;
  return (m % required_block_size == 0) && 
         (n % required_block_size == 0) && 
         (k % required_block_size == 0);
}

} // namespace at::cuda::tunable 