#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>

namespace c10::cuda::CUDACachingAllocator {

// Keep this for backwards compatibility
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return c10::CachingAllocator::AllocatorConfig::instance().max_split_size();
  }
  static double garbage_collection_threshold() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .garbage_collection_threshold();
  }

  static bool expandable_segments() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .use_expandable_segments();
  }

  static bool release_lock_on_cudamalloc() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .use_release_lock_on_device_malloc();
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .pinned_use_device_host_register();
  }

  static size_t pinned_num_register_threads() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .pinned_num_register_threads();
  }

  static bool pinned_use_background_threads() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .pinned_use_background_threads();
  }

  static size_t pinned_max_register_threads() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .pinned_max_register_threads();
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static std::vector<size_t> roundup_power2_divisions() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .roundup_power2_divisions();
  }

  static size_t max_non_split_rounding_size() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .max_non_split_rounding_size();
  }

  static std::string last_allocator_settings() {
    return c10::CachingAllocator::AllocatorConfig::instance()
        .last_allocator_settings();
  }

  static CUDAAllocatorConfig& instance() {
    static CUDAAllocatorConfig* s_instance = ([]() {
      auto inst = new CUDAAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
#ifdef USE_ROCM
      // convenience for ROCm users, allow alternative HIP token
      if (!env) {
        env = getenv("PYTORCH_HIP_ALLOC_CONF");
      }
#endif
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CUDAAllocatorConfig() = default;
};

// Keep this for backwards compatibility
C10_CUDA_API void setAllocatorSettings(const std::string& env);

} // namespace c10::cuda::CUDACachingAllocator
