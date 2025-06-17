#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace c10::cuda::CUDACachingAllocator {

// Keep this for backwards compatibility
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return CachingAllocator::AllocatorConfig::max_split_size();
  }
  static double garbage_collection_threshold() {
    return CachingAllocator::AllocatorConfig::garbage_collection_threshold();
  }

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

  static size_t pinned_num_register_threads() {
    return CachingAllocator::AllocatorConfig::pinned_num_register_threads();
  }

  static bool pinned_use_background_threads() {
    return CachingAllocator::AllocatorConfig::pinned_use_background_threads();
  }

  static size_t pinned_max_register_threads() {
    return CachingAllocator::AllocatorConfig::pinned_max_register_threads();
  }

  static size_t roundup_power2_divisions(size_t size) {
    return CachingAllocator::AllocatorConfig::roundup_power2_divisions(size);
  }

  static std::vector<size_t> roundup_power2_divisions() {
    return CachingAllocator::AllocatorConfig::roundup_power2_divisions();
  }

  static size_t max_non_split_rounding_size() {
    return CachingAllocator::AllocatorConfig::max_non_split_rounding_size();
  }

  static std::string last_allocator_settings() {
    return CachingAllocator::AllocatorConfig::last_allocator_settings();
  }

 private:
  CUDAAllocatorConfig() = default;
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
