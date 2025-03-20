#pragma once

#include <c10/core/DeviceType.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace c10::CachingAllocator {

// Environment config parser for Allocator
class C10_API AllocatorConfig {
 public:
  static AllocatorConfig& instance();

  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

 private:
  std::atomic<size_t> m_max_split_size;
  std::atomic<size_t> m_max_non_split_rounding_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<size_t> m_pinned_num_register_threads;
  std::atomic<bool> m_expandable_segments;
  std::atomic<bool> m_release_lock_on_cudamalloc;
  std::atomic<bool> m_pinned_use_host_register;
  std::atomic<bool> m_pinned_use_background_threads;
  std::string m_last_allocator_settings;
  std::mutex m_last_allocator_settings_mutex;
};

C10_API void SetAllocatorConfig(
    at::DeviceType t,
    AllocatorConfig* allocator_config);
C10_API AllocatorConfig* GetAllocatorConfig(const at::DeviceType& t);

} // namespace c10::CachingAllocator
