#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

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
    return c10::CachingAllocator::AcceleratorAllocatorConfig::max_split_size();
  }
  static double garbage_collection_threshold() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        garbage_collection_threshold();
  }

  static bool expandable_segments() {
    bool enabled = c10::CachingAllocator::AcceleratorAllocatorConfig::
        use_expandable_segments();
#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
    if (enabled) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return enabled;
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

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return instance().m_pinned_use_cuda_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static bool pinned_use_background_threads() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        pinned_use_background_threads();
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
  static size_t roundup_power2_divisions(size_t size) {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions(size);
  }

  static std::vector<size_t> roundup_power2_divisions() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions();
  }

  static size_t max_non_split_rounding_size() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        max_non_split_rounding_size();
  }

  static std::string last_allocator_settings() {
    return c10::CachingAllocator::getAllocatorSettings();
  }

  static bool use_async_allocator() {
    return instance().m_use_async_allocator;
  }

  // Use `Construct On First Use Idiom` to avoid `Static Initialization Order`
  // issue.
  static const std::unordered_set<std::string>& getKeys() {
    static std::unordered_set<std::string> keys{
        "backend",
        // keep BC for Rocm: `cuda` -> `cud` `a`, to avoid hipify issues
        // NOLINTBEGIN(bugprone-suspicious-missing-comma,-warnings-as-errors)
        "release_lock_on_cud"
        "amalloc",
        "pinned_use_cud"
        "a_host_register",
        // NOLINTEND(bugprone-suspicious-missing-comma,-warnings-as-errors)
        "release_lock_on_hipmalloc",
        "pinned_use_hip_host_register",
        "pinned_num_register_threads"};
    return keys;
  }

  static CUDAAllocatorConfig& instance() {
    static CUDAAllocatorConfig* s_instance = ([]() {
      auto inst = new CUDAAllocatorConfig();
      auto env = c10::utils::get_env("PYTORCH_ALLOC_CONF");
      if (!env.has_value()) {
        // For backward compatibility, check for the old environment variable
        // PYTORCH_CUDA_ALLOC_CONF.
        env = c10::utils::get_env("PYTORCH_CUDA_ALLOC_CONF");
      }
#ifdef USE_ROCM
      // convenience for ROCm users, allow alternative HIP token
      if (!env.has_value()) {
        env = c10::utils::get_env("PYTORCH_HIP_ALLOC_CONF");
      }
#endif
      if (env.has_value()) {
        inst->parseArgs(env.value());
      }
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const std::string& env);

 private:
  CUDAAllocatorConfig() = default;

  size_t parseAllocatorConfig(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedUseCudaHostRegister(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedNumRegisterThreads(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);

  std::atomic<size_t> m_pinned_num_register_threads{1};
  std::atomic<Expandable_Segments_Handle_Type> m_expandable_segments_handle_type
#if CUDA_VERSION >= 12030
      {Expandable_Segments_Handle_Type::UNSPECIFIED};
#else
      {Expandable_Segments_Handle_Type::POSIX_FD};
#endif
  std::atomic<bool> m_release_lock_on_cudamalloc{false};
  std::atomic<bool> m_pinned_use_cuda_host_register{false};
  std::atomic<bool> m_use_async_allocator{false};
  std::atomic<bool> m_is_allocator_loaded{false};
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
