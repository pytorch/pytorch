#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>

namespace c10::cuda::CUDACachingAllocator {

// Keep this for backwards compatibility
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return getAllocatorConfig().max_split_size();
  }
  static double garbage_collection_threshold() {
    return getAllocatorConfig().garbage_collection_threshold();
  }

  static bool expandable_segments() {
    return getAllocatorConfig().use_expandable_segments();
  }

  static bool release_lock_on_cudamalloc() {
    return getAllocatorConfig().use_release_lock_on_device_malloc();
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return getAllocatorConfig().pinned_use_device_host_register();
  }

  static size_t pinned_num_register_threads() {
    return getAllocatorConfig().pinned_num_register_threads();
  }

  static bool pinned_use_background_threads() {
    return getAllocatorConfig().pinned_use_background_threads();
  }

  static size_t pinned_max_register_threads() {
    return getAllocatorConfig().pinned_max_register_threads();
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size) {
    return getAllocatorConfig().roundup_power2_divisions(size);
  }

  static std::vector<size_t> roundup_power2_divisions() {
    return getAllocatorConfig().roundup_power2_divisions();
  }

  static size_t max_non_split_rounding_size() {
    return getAllocatorConfig().max_non_split_rounding_size();
  }

  static std::string last_allocator_settings() {
    return getAllocatorConfig().last_allocator_settings();
  }

  void parseArgs(const char* env) {
    getAllocatorConfig().parseArgs(env);
  }

 private:
  CUDAAllocatorConfig() = default;

  static c10::CachingAllocator::AllocatorConfig& getAllocatorConfig() {
    return c10::CachingAllocator::getAllocatorConfig();
  }
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
