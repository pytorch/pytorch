#pragma once

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace c10::CachingAllocator {

// "large" allocations may be packed in 20 MiB blocks
const size_t kLargeBuffer = 20971520;

// Environment config parser for Allocator
class C10_API AllocatorConfig {
 public:
  static AllocatorConfig& instance();

  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static size_t max_non_split_rounding_size() {
    return instance().m_max_non_split_rounding_size;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // on CUDA device using env variable:
  // PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static std::vector<size_t> roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
    return instance().m_expandable_segments;
  }

  static bool release_lock_on_device_malloc() {
    return instance().m_release_lock_on_device_malloc;
  }

  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(
        instance().m_last_allocator_settings_mutex);
    return instance().m_last_allocator_settings;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_host_register() {
    return instance().m_pinned_use_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static bool pinned_use_background_threads() {
    return instance().m_pinned_use_background_threads;
  }

 private:
  // The maximum block size that is allowed to be split. Default is
  // std::numeric_limits<size_t>::max()
  std::atomic<size_t> m_max_split_size;
  // The maximum allowable extra size of a memory block without requiring
  // splitting when searching for a free block. Default is kLargeBuffer
  std::atomic<size_t> m_max_non_split_rounding_size;
  // Used to store how memory allocations of different sizes should be rounded
  // up to the nearest power of 2 divisions.
  std::vector<size_t> m_roundup_power2_divisions;
  // The threshold that triggers garbage collection when the ratio of used
  // memory to maximum allowed memory exceeds this value. Default is 0.0
  std::atomic<double> m_garbage_collection_threshold;
  // A flag to enable expandable segments feature. The default value is false.
  std::atomic<bool> m_expandable_segments;
  // A flag to release the lock on device malloc. The default value is false.
  std::atomic<bool> m_release_lock_on_device_malloc;
  // Record the last allocator config environment setting.
  std::mutex m_last_allocator_settings_mutex;
  std::string m_last_allocator_settings;
  // A flag that determines whether to register a CPU allocation for use by
  // device. The default value is false.
  std::atomic<bool> m_pinned_use_host_register;
  // The number of threads to parallelize to register a CPU allocation to reduce
  // the overall time. Default is 1.
  std::atomic<size_t> m_pinned_num_register_threads;
  // A flag to enable background thread for processing events. The default value
  // is false.
  std::atomic<bool> m_pinned_use_background_threads;
};

C10_API void SetAllocatorConfig(
    at::DeviceType t,
    AllocatorConfig* allocator_config);
C10_API AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t);

} // namespace c10::CachingAllocator
