#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Deprecated.h>
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
  static bool expandable_segments() {
    bool enabled = c10::CachingAllocator::AcceleratorAllocatorConfig::
        use_expandable_segments();
#if !defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    (!defined(USE_ROCM) || (ROCM_VERSION < 70000))
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

  static bool graph_capture_record_stream_reuse() {
    return instance().m_graph_capture_record_stream_reuse;
  }

  static double per_process_memory_fraction() {
    return instance().m_per_process_memory_fraction;
  }

  // When enabled, throws OOM error before calling cudaMalloc if the allocation
  // would likely fail due to insufficient memory. This provides early failure
  // with clear error messages instead of letting cudaMalloc fail.
  static bool throw_on_cudamalloc_oom() {
    return instance().m_throw_on_cudamalloc_oom;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return instance().m_pinned_use_cuda_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
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

  static bool pinned_free_catch_all() {
    return instance().m_pinned_free_catch_all;
  }

  static size_t max_non_split_rounding_size() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        max_non_split_rounding_size();
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
      // Note: keep the parsing order and logic stable to avoid potential
      // performance regressions in internal tests.
      if (!env.has_value()) {
        env = c10::utils::get_env("PYTORCH_ALLOC_CONF");
      }
      if (env.has_value()) {
        inst->parseArgs(env.value());
      }
      return inst;
    })();
    return *s_instance;
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
        "graph_capture_record_stream_reuse",
        "pinned_reserve_segment_size_mb",
        "pinned_num_register_threads",
        "per_process_memory_fraction",
        "pinned_free_catch_all",
        "throw_on_cudamalloc_oom"};
    return keys;
  }

  void parseArgs(const std::string& env);

 private:
  CUDAAllocatorConfig() = default;

  size_t parseAllocatorConfig(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i,
      bool& used_cudaMallocAsync);
  size_t parsePinnedUseCudaHostRegister(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedNumRegisterThreads(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedReserveSegmentSize(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseGraphCaptureRecordStreamReuse(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePerProcessMemoryFraction(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedFreeCatchAll(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseThrowOnCudaMallocOom(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);

  std::atomic<size_t> m_pinned_num_register_threads{1};
  std::atomic<size_t> m_pinned_reserve_segment_size_mb{0};
  std::atomic<Expandable_Segments_Handle_Type> m_expandable_segments_handle_type
#if CUDA_VERSION >= 12030
      {Expandable_Segments_Handle_Type::UNSPECIFIED};
#else
      {Expandable_Segments_Handle_Type::POSIX_FD};
#endif
  std::atomic<bool> m_release_lock_on_cudamalloc{false};
  std::atomic<bool> m_pinned_use_cuda_host_register{false};
  std::atomic<bool> m_graph_capture_record_stream_reuse{false};
  std::atomic<double> m_per_process_memory_fraction{1.0};
  std::atomic<bool> m_pinned_free_catch_all{false};
  // When true, throw OOM error before calling cudaMalloc if allocation would
  // fail
  std::atomic<bool> m_throw_on_cudamalloc_oom{false};
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
