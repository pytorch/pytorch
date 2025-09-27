#pragma once

#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace c10::cuda::CUDACachingAllocator {

enum class Expandable_Segments_Handle_Type : int {
  UNSPECIFIED = 0,
  POSIX_FD = 1,
  FABRIC_HANDLE = 2,
};

// Environment config parser
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

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

  static bool graph_capture_record_stream_reuse() {
    return instance().m_graph_capture_record_stream_reuse;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return instance().m_pinned_use_cuda_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static bool pinned_use_background_threads() {
    return instance().m_pinned_use_background_threads;
  }

  static size_t pinned_reserve_segment_size_mb() {
    return instance().m_pinned_reserve_segment_size_mb;
  }

  static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static std::vector<size_t> roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }

  static size_t max_non_split_rounding_size() {
    return instance().m_max_non_split_rounding_size;
  }

  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(
        instance().m_last_allocator_settings_mutex);
    return instance().m_last_allocator_settings;
  }

  static CUDAAllocatorConfig& instance() {
    static CUDAAllocatorConfig* s_instance = ([]() {
      auto inst = new CUDAAllocatorConfig();
      auto env = c10::utils::get_env("PYTORCH_CUDA_ALLOC_CONF");
#ifdef USE_ROCM
      // convenience for ROCm users, allow alternative HIP token
      if (!env.has_value()) {
        env = c10::utils::get_env("PYTORCH_HIP_ALLOC_CONF");
      }
#endif
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const std::optional<std::string>& env);

 private:
  CUDAAllocatorConfig();

  static void lexArgs(const std::string& env, std::vector<std::string>& config);
  static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseMaxNonSplitRoundingSize(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
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
  size_t parsePinnedReserveSegmentSize(
      const std::vector<std::string>& config,
      size_t i);
  size_t parsePinnedUseBackgroundThreads(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseGraphCaptureRecordStreamReuse(
      const std::vector<std::string>& config,
      size_t i);

  std::atomic<size_t> m_max_split_size;
  std::atomic<size_t> m_max_non_split_rounding_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<size_t> m_pinned_num_register_threads;
  std::atomic<size_t> m_pinned_reserve_segment_size_mb;
  std::atomic<bool> m_expandable_segments;
  std::atomic<Expandable_Segments_Handle_Type>
      m_expandable_segments_handle_type;
  std::atomic<bool> m_release_lock_on_cudamalloc;
  std::atomic<bool> m_pinned_use_cuda_host_register;
  std::atomic<bool> m_graph_capture_record_stream_reuse;
  std::atomic<bool> m_pinned_use_background_threads;
  std::string m_last_allocator_settings;
  std::mutex m_last_allocator_settings_mutex;
};

// General caching allocator utilities
C10_CUDA_API void setAllocatorSettings(const std::string& env);

} // namespace c10::cuda::CUDACachingAllocator
