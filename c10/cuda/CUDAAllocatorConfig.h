#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace c10::cuda::CUDACachingAllocator {

// Keep this for backwards compatibility
class C10_CUDA_API CUDAAllocatorConfig
    : public CachingAllocator::AllocatorConfig {
 public:
  static bool expandable_segments() {
    return CachingAllocator::AllocatorConfig::use_expandable_segments();
  }

  static bool release_lock_on_cudamalloc() {
    return CachingAllocator::AllocatorConfig::
        use_release_lock_on_device_malloc();
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return CachingAllocator::AllocatorConfig::pinned_use_device_host_register();
  }

};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
