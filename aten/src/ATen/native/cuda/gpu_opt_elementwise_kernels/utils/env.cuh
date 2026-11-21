#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include <cuda_runtime_api.h>

namespace at::native {

struct GPUDeviceCapabilities {
  bool supports_sm90{false};
  int64_t sm_count{0};
  std::array<int64_t, 3> max_grid_size{0, 0, 0};
};

// Cache only devices that have actually been selected by the caller.  This
// avoids probing unrelated cards in heterogeneous multi-device processes.
static inline const GPUDeviceCapabilities* current_gpu_device_capabilities() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }

  static std::mutex cache_mutex;
  static std::map<int, GPUDeviceCapabilities> cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto found = cache.find(device);
  if (found != cache.end()) {
    return &found->second;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return nullptr;
  }
  GPUDeviceCapabilities capabilities;
  capabilities.supports_sm90 = prop.major >= 9;
  capabilities.sm_count = prop.multiProcessorCount;
  for (int axis = 0; axis < 3; ++axis) {
    capabilities.max_grid_size[axis] = prop.maxGridSize[axis];
  }
  return &cache.emplace(device, capabilities).first->second;
}

static inline bool is_sm90_or_newer_device() {
  const GPUDeviceCapabilities* capabilities = current_gpu_device_capabilities();
  return capabilities != nullptr && capabilities->supports_sm90;
}

// Returns zero when the current device cannot be queried; callers treat that
// as an ineligible launch rather than constructing an unchecked dim3 value.
static inline int64_t gpu_max_grid_size(int axis) {
  const GPUDeviceCapabilities* capabilities = current_gpu_device_capabilities();
  if (capabilities == nullptr || axis < 0 || axis >= 3) {
    return 0;
  }
  return capabilities->max_grid_size[axis];
}

// Returns zero when the current device cannot be queried; callers use it
// to derive occupancy thresholds and treat zero as "keep the default".
static inline int64_t gpu_sm_count() {
  const GPUDeviceCapabilities* capabilities = current_gpu_device_capabilities();
  return capabilities != nullptr ? capabilities->sm_count : 0;
}

// Empirical sweet cap on host-side grid block counts (2^16 - 1). 65535 is
// both the historical gridDim limit and the measured sweet spot: below it
// per-block scheduling stays efficient, above it the per-block overhead
// dominates and the wider-grid elementwise variants regress by 1.3-3.6x
// (measured on the 2_2/3_1/3_2 families at 4M-64M). Tune this single macro
// instead of the per-site min() expressions.
#define GPU_GRID_SWEET_CAP 65535

// Convenience clamp for older call sites: the device's gridDim limit
// tightened by the empirical sweet cap above. New code spells the two
// bounds out at each gate instead - std::min<int64_t>(gpu_max_grid_size(
// axis), GPU_GRID_SWEET_CAP) - so the hardware limit and the performance
// sweet spot stay distinguishable per site.
static inline int64_t gpu_grid_cap(int axis) {
  return std::min<int64_t>(gpu_max_grid_size(axis), GPU_GRID_SWEET_CAP);
}

// Enable the gpu noncontiguous-specialized kernel path (off by default):
// unless PYTORCH_ENABLE_GPU_ELEMENTWISE_OPT is set to a non-zero, non-False
// value, noncontiguous calls fall through to the legacy path. The environment
// switch is process-wide and cached independently from the current device.
static inline bool elementwise_gpu_enabled() {
  static const bool env_enabled = []() {
    const char* v = std::getenv("PYTORCH_ENABLE_GPU_ELEMENTWISE_OPT");
    return v != nullptr && v[0] != '\0' && v[0] != '0' && v[0] != 'F' && v[0] != 'f';
  }();
  return env_enabled && is_sm90_or_newer_device();
}

static inline const std::string& local_rank() {
  static const std::string rank = []() {
    if (const char* value = std::getenv("LOCAL_RANK")) {
      return std::string(value);
    }
    if (const char* value = std::getenv("RANK")) {
      return std::string(value);
    }
    return std::string("0");
  }();
  return rank;
}

} // namespace at::native
