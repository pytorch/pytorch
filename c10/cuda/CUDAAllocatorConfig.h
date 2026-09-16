#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/flat_hash_map.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

namespace c10::cuda::CUDACachingAllocator {

enum class Expandable_Segments_Handle_Type : int {
  UNSPECIFIED = 0,
  POSIX_FD = 1,
  FABRIC_HANDLE = 2,
};

// A per-segment virtual-address reserve size, expressed either as a fraction of
// a device's total memory (e.g. 0.5) or as an absolute number of GiB (config
// suffix 'G', e.g. 40G). Resolved to an absolute byte count against a device's
// total memory.
struct ExpandableSegmentReserveSpec {
  bool is_fraction = true;
  double value = 0.0;

  size_t resolveBytes(size_t total_global_mem) const {
    double bytes = is_fraction ? value * static_cast<double>(total_global_mem)
                               : value * static_cast<double>(size_t{1} << 30);
    // Saturate before narrowing: converting a double that exceeds SIZE_MAX to
    // size_t is undefined behavior, and a pathological config value (e.g.
    // expandable_segments_reserve:1e20G) can produce such a double.
    // clamp_reserve_bytes() bounds the reserve to the full reserve, so a
    // saturated value here can never reach numSegments().
    if (bytes >= static_cast<double>(std::numeric_limits<size_t>::max())) {
      return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(bytes);
  }
};

// A consistent snapshot of the reserve decision for a class, read under a
// single lock so a concurrent config re-parse cannot compose an inconsistent
// view.
struct ExpandableSegmentReserveDecision {
  // nullopt => no override configured; caller keeps the full historical
  // reserve.
  std::optional<size_t> reserve_bytes;
  bool class_known = false; // the class had an explicit per-class entry
};

// Environment config parser
class C10_CUDA_API CUDAAllocatorConfig {
 public:
  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::max_split_size() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::max_split_size() instead.")
  static size_t max_split_size() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::max_split_size();
  }

  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::garbage_collection_threshold() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::garbage_collection_threshold() instead.")
  static double garbage_collection_threshold() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        garbage_collection_threshold();
  }

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

  // Single-lock snapshot of the reserve decision for reserve_class, so the
  // segment-creation path composes (reserve, class_known) consistently even if
  // setAllocatorSettings() re-parses concurrently.
  static ExpandableSegmentReserveDecision expandable_segments_reserve_decision(
      const std::string& reserve_class,
      size_t total_global_mem) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.m_reserve_mutex);
    ExpandableSegmentReserveDecision d;
    auto it = self.m_expandable_segments_reserve_by_class.find(reserve_class);
    if (it != self.m_expandable_segments_reserve_by_class.end()) {
      d.reserve_bytes = it->second.resolveBytes(total_global_mem);
      d.class_known = true;
    } else if (self.m_expandable_segments_reserve_set) {
      d.reserve_bytes =
          self.m_expandable_segments_reserve.resolveBytes(total_global_mem);
    }
    return d;
  }

  // Final per-segment reserve for a decision snapshot, given the device's full
  // historical reserve (9/8 of total memory). Returns full_reserve unchanged
  // when no override is configured, so the untagged path is byte-for-byte
  // identical. Otherwise the configured reserve is capped at full_reserve: no
  // single allocation can exceed physical memory, so a larger reservation is
  // never needed, and the cap also keeps a saturated reserve (a pathological
  // expandable_segments_reserve) from overflowing numSegments(). Pure, so it is
  // unit-testable without a device.
  static size_t clamp_reserve_bytes(
      const ExpandableSegmentReserveDecision& decision,
      size_t full_reserve) {
    if (!decision.reserve_bytes.has_value()) {
      return full_reserve;
    }
    return std::min(*decision.reserve_bytes, full_reserve);
  }

  // Registers a code-side default reserve for a class, letting serving layers
  // opt their stream classes into a downsized reserve by default. Prefer the
  // setDefaultExpandableSegmentReserveFractionForClass() free function over
  // calling this directly. Precedence is resolved at call time: this is a no-op
  // if a global reserve (expandable_segments_reserve) has been set, or if
  // reserve_class already has an explicit per-class entry. Call only after
  // config is parsed (touching instance() forces that); a later runtime
  // re-parse that sets only a global reserve does not retroactively remove an
  // already-seeded default.
  static void set_default_reserve_for_class(
      const std::string& reserve_class,
      ExpandableSegmentReserveSpec spec) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.m_reserve_mutex);
    if (self.m_expandable_segments_reserve_set) {
      return;
    }
    self.m_expandable_segments_reserve_by_class.emplace(reserve_class, spec);
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

  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_background_threads() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::pinned_use_background_threads() instead.")
  static bool pinned_use_background_threads() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        pinned_use_background_threads();
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

  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::roundup_power2_divisions() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::roundup_power2_divisions() instead.")
  static size_t roundup_power2_divisions(size_t size) {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions(size);
  }

  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::roundup_power2_divisions() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::roundup_power2_divisions() instead.")
  static std::vector<size_t> roundup_power2_divisions() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions();
  }

  static size_t max_non_split_rounding_size() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        max_non_split_rounding_size();
  }

  C10_DEPRECATED_MESSAGE(
      "c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::last_allocator_settings() is deprecated. Please use c10::CachingAllocator::AcceleratorAllocatorConfig::last_allocator_settings() instead.")
  static std::string last_allocator_settings() {
    return c10::CachingAllocator::getAllocatorSettings();
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
        "throw_on_cudamalloc_oom",
        "expandable_segments_reserve",
        "expandable_segments_reserve_by_class"};
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
  size_t parseExpandableSegmentsReserve(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseExpandableSegmentsReserveByClass(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  // Parses a reserve value ("<fraction>" or "<n>G"). Returns nullopt on a
  // malformed or non-positive value (logged, non-fatal) so callers keep
  // defaults.
  static std::optional<ExpandableSegmentReserveSpec> parseReserveSpec(
      const std::string& token);

  std::atomic<size_t> m_pinned_num_register_threads{1};
  std::atomic<size_t> m_pinned_reserve_segment_size_mb{0};
  // UNSPECIFIED resolves to FABRIC where supported and POSIX_FD otherwise, so a
  // separate POSIX_FD default for older CUDA is redundant.
  std::atomic<Expandable_Segments_Handle_Type>
      m_expandable_segments_handle_type{
          Expandable_Segments_Handle_Type::UNSPECIFIED};
  std::atomic<bool> m_release_lock_on_cudamalloc{false};
  std::atomic<bool> m_pinned_use_cuda_host_register{false};
  std::atomic<bool> m_graph_capture_record_stream_reuse{false};
  std::atomic<double> m_per_process_memory_fraction{1.0};
  std::atomic<bool> m_pinned_free_catch_all{false};
  // When true, throw OOM error before calling cudaMalloc if allocation would
  // fail
  std::atomic<bool> m_throw_on_cudamalloc_oom{false};
  // Per-stream expandable-segment reserve config. Parsed once at init (like the
  // rest of PYTORCH_CUDA_ALLOC_CONF); read on the segment-creation path.
  // Default reserve, applied to untagged streams and to classes with no
  // override. 1.125 mirrors the historical "1 1/8" reserve that the
  // ExpandableSegment ctor computes as (total + total/8); it is only consulted
  // when explicitly set via config, so the untagged path stays byte-for-byte
  // identical when unset.
  ExpandableSegmentReserveSpec m_expandable_segments_reserve{
      /*is_fraction=*/true,
      1.125};
  bool m_expandable_segments_reserve_set{false};
  ska::flat_hash_map<std::string, ExpandableSegmentReserveSpec>
      m_expandable_segments_reserve_by_class;
  // Guards the reserve fields above: they are read on the segment-creation path
  // and can be rewritten at runtime by a setAllocatorSettings() re-parse.
  std::mutex m_reserve_mutex;
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
