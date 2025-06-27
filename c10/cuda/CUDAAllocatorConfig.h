#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace c10::cuda::CUDACachingAllocator {

class C10_CUDA_API CUDAAllocatorConfig {
 public:
  static bool expandable_segments() {
#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
}

  static Expandable_Segments_Handle_Type expandable_segments_handle_type() {
    return instance().m_expandable_segments_handle_type;
  }

  static void set_expandable_segments_handle_type(
      Expandable_Segments_Handle_Type handle_type) {
    instance().m_expandable_segments_handle_type = handle_type;
  }

  static bool release_lock_on_cudamalloc() {
    return instance().m_release_lock_on_cudamalloc;
  }

  static bool pinned_use_cuda_host_register() {
    return instance().m_pinned_use_cuda_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

 private:
  CUDAAllocatorConfig();

  static void lexArgs(const std::string& env, std::vector<std::string>& config);
  static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);
  size_t parsePinnedUseCudaHostRegister(
      const std::vector<std::string>& config,
      size_t i);
  size_t parsePinnedNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);

  std::atomic<size_t> m_pinned_num_register_threads;
  std::atomic<Expandable_Segments_Handle_Type>
      m_expandable_segments_handle_type;
  std::atomic<bool> m_release_lock_on_cudamalloc;
  std::atomic<bool> m_pinned_use_cuda_host_register;
  std::string m_last_allocator_settings;
  std::mutex m_last_allocator_settings_mutex;
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
