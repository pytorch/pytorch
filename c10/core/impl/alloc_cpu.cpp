#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/alignment.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <c10/util/irange.h>
#include <c10/util/numa.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>

#ifdef USE_MIMALLOC
#include <mimalloc-stats.h>
#include <mimalloc.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

// TODO: rename flags to C10
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU")

// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU")

namespace c10 {

namespace {

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  auto int64_count = num / sizeof(kJunkPattern64);
  auto remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (const auto i : c10::irange(int64_count)) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
inline bool is_thp_alloc_enabled() {
  static bool value = [&] {
    auto env = c10::utils::check_env("THP_MEM_ALLOC_ENABLE");
    return env.has_value() ? env.value() : 0;
  }();
  return value;
}

inline bool is_thp_alloc(size_t nbytes) {
  // enable thp (transparent huge pages) for larger buffers
  return (is_thp_alloc_enabled() && (nbytes >= gAlloc_threshold_thp));
}

#elif !defined(__ANDROID__) && !defined(_MSC_VER)
constexpr size_t c10_compute_alignment(size_t /*nbytes*/) {
  return gAlignment;
}

constexpr bool is_thp_alloc([[maybe_unused]] size_t nbytes) {
  return false;
}
#endif

#ifdef USE_MIMALLOC
mi_option_t parse_mimalloc_option_name(const std::string& name) {
  static const std::unordered_map<std::string, mi_option_t> options = {
      // stable options
      {"show_errors", mi_option_show_errors},
      {"show_stats", mi_option_show_stats},
      {"verbose", mi_option_verbose},
      {"max_errors", mi_option_max_errors},
      {"max_warnings", mi_option_max_warnings},

      // advanced options
      {"reserve_huge_os_pages", mi_option_reserve_huge_os_pages},
      {"reserve_huge_os_pages_at", mi_option_reserve_huge_os_pages_at},
      {"reserve_os_memory", mi_option_reserve_os_memory},
      {"allow_large_os_pages", mi_option_allow_large_os_pages},
      {"purge_decommits", mi_option_purge_decommits},
      {"arena_reserve", mi_option_arena_reserve},
      {"os_tag", mi_option_os_tag},
      {"retry_on_oom", mi_option_retry_on_oom},
      {"generic_collect", mi_option_generic_collect},
      {"allow_thp", mi_option_allow_thp},

      // guard pages
      {"guarded_min", mi_option_guarded_min},
      {"guarded_max", mi_option_guarded_max},
      {"guarded_precise", mi_option_guarded_precise},
      {"guarded_sample_rate", mi_option_guarded_sample_rate},
      {"guarded_sample_seed", mi_option_guarded_sample_seed},

      // experimental options
      {"eager_commit", mi_option_eager_commit},
      {"eager_commit_delay", mi_option_eager_commit_delay},
      {"arena_eager_commit", mi_option_arena_eager_commit},
      {"abandoned_page_purge", mi_option_abandoned_page_purge},
      {"purge_delay", mi_option_purge_delay},
      {"use_numa_nodes", mi_option_use_numa_nodes},
      {"disallow_os_alloc", mi_option_disallow_os_alloc},
      {"max_segment_reclaim", mi_option_max_segment_reclaim},
      {"destroy_on_exit", mi_option_destroy_on_exit},
      {"arena_purge_mult", mi_option_arena_purge_mult},
      {"abandoned_reclaim_on_free", mi_option_abandoned_reclaim_on_free},
      {"purge_extend_delay", mi_option_purge_extend_delay},
      {"disallow_arena_alloc", mi_option_disallow_arena_alloc},
      {"visit_abandoned", mi_option_visit_abandoned},
      {"target_segments_per_thread", mi_option_target_segments_per_thread},
  };

  auto it = options.find(name);
  TORCH_CHECK(it != options.end(), "Unknown mimalloc option: ", name);
  return it->second;
}

#endif
} // namespace

#if defined(__linux__) && !defined(__ANDROID__)
size_t c10_compute_alignment(size_t nbytes) {
  static const auto pagesize = sysconf(_SC_PAGESIZE);
  // for kernels that don't provide page size, default it to 4K
  const size_t thp_alignment = (pagesize < 0 ? gPagesize : pagesize);
  return (is_thp_alloc(nbytes) ? thp_alignment : gAlignment);
}
#endif

void* alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
      ((ptrdiff_t)nbytes) >= 0,
      "alloc_cpu() seems to have been called with negative number: ",
      nbytes);

  void* data = nullptr;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(USE_MIMALLOC)
  data = mi_malloc_aligned(nbytes, gAlignment);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#else
  int err = posix_memalign(&data, c10_compute_alignment(nbytes), nbytes);
  CAFFE_ENFORCE(
      err == 0,
      "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
      nbytes,
      " bytes. Error code ",
      err,
      " (",
      c10::utils::str_error(err),
      ")");
  if (is_thp_alloc(nbytes)) {
#ifdef __linux__
    // MADV_HUGEPAGE advise is available only for linux.
    // general posix compliant systems can check POSIX_MADV_SEQUENTIAL advise.
    int ret = madvise(data, nbytes, MADV_HUGEPAGE);
    if (ret != 0) {
      TORCH_WARN_ONCE(
          "thp madvise for HUGEPAGE failed with ",
          c10::utils::str_error(errno));
    }
#endif
  }
#endif

  // move data to a thread's NUMA node
  NUMAMove(data, nbytes, GetCurrentNUMANode());
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    memset_junk(data, nbytes);
  }

  return data;
}

void free_cpu(void* data) {
#ifdef USE_MIMALLOC
  mi_free(data);
#elif defined(_MSC_VER)
  _aligned_free(data);
#else
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(data);
#endif
}

bool is_mimalloc_enabled() {
#ifdef USE_MIMALLOC
  return true;
#else
  return false;
#endif
}

std::string get_mimalloc_stats_json() {
#ifdef USE_MIMALLOC
  char* stats = mi_stats_get_json(0, nullptr);
  TORCH_CHECK(stats != nullptr, "Failed to get mimalloc stats");
  std::string result(stats);
  mi_free(stats);
  return result;
#else
  TORCH_CHECK(false, "mimalloc is not enabled in this build");
#endif
}

void reset_mimalloc_stats() {
#ifdef USE_MIMALLOC
  mi_stats_reset();
#else
  TORCH_CHECK(false, "mimalloc is not enabled in this build");
#endif
}

void set_mimalloc_option(const std::string& name, int64_t value) {
#ifdef USE_MIMALLOC
#if INT64_MAX > LONG_MAX
  // Mimalloc's options are passed as long which may be 32 bit on some
  // platforms whereas we prefer a platform-independent int64_t:
  TORCH_CHECK(
      value >= std::numeric_limits<long>::min() &&
          value <= std::numeric_limits<long>::max(),
      "mimalloc option value out of range: ",
      value);
#endif
  mi_option_set(parse_mimalloc_option_name(name), static_cast<long>(value));
#else
  TORCH_CHECK(false, "mimalloc is not enabled in this build");
#endif
}

int64_t get_mimalloc_option(const std::string& name) {
#ifdef USE_MIMALLOC
  return mi_option_get(parse_mimalloc_option_name(name));
#else
  TORCH_CHECK(false, "mimalloc is not enabled in this build");
#endif
}

#ifdef USE_MIMALLOC_ON_MKL
namespace mi_malloc_wrapper {
void* c10_mi_malloc(size_t size) {
  return mi_malloc(size);
}

void* c10_mi_calloc(size_t count, size_t size) {
  return mi_calloc(count, size);
}

void* c10_mi_realloc(void* p, size_t newsize) {
  return mi_realloc(p, newsize);
}

void* c10_mi_malloc_aligned(size_t size, size_t alignment) {
  return mi_malloc_aligned(size, alignment);
}

void c10_mi_free(void* p) {
  mi_free(p);
}
} // namespace mi_malloc_wrapper
#endif
} // namespace c10
