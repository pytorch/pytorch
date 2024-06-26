#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/alignment.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/numa.h>

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

// TODO: rename flags to C10
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU");

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
static inline bool is_thp_alloc_enabled() {
  static bool value = [&] {
    const char* ptr = std::getenv("THP_MEM_ALLOC_ENABLE");
    return ptr != nullptr ? std::atoi(ptr) : 0;
  }();
  return value;
}

inline size_t c10_compute_alignment(size_t nbytes) {
  static const auto pagesize = sysconf(_SC_PAGESIZE);
  // for kernels that don't provide page size, default it to 4K
  const size_t thp_alignment = (pagesize < 0 ? gPagesize : pagesize);
  return (is_thp_alloc_enabled() ? thp_alignment : gAlignment);
}

inline bool is_thp_alloc(size_t nbytes) {
  // enable thp (transparent huge pages) for larger buffers
  return (is_thp_alloc_enabled() && (nbytes >= gAlloc_threshold_thp));
}
#elif !defined(__ANDROID__) && !defined(_MSC_VER)
constexpr size_t c10_compute_alignment(C10_UNUSED size_t nbytes) {
  return gAlignment;
}

constexpr bool is_thp_alloc(C10_UNUSED size_t nbytes) {
  return false;
}
#endif
} // namespace

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

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* data;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(_MSC_VER)
#ifdef USE_MIMALLOC
  data = mi_malloc_aligned(nbytes, gAlignment);
#else
  data = _aligned_malloc(nbytes, gAlignment);
#endif
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
      strerror(err),
      ")");
  if (is_thp_alloc(nbytes)) {
#ifdef __linux__
    // MADV_HUGEPAGE advise is available only for linux.
    // general posix compliant systems can check POSIX_MADV_SEQUENTIAL advise.
    int ret = madvise(data, nbytes, MADV_HUGEPAGE);
    if (ret != 0) {
      TORCH_WARN_ONCE("thp madvise for HUGEPAGE failed with ", strerror(errno));
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
#ifdef _MSC_VER
#ifdef USE_MIMALLOC
  mi_free(data);
#else
  _aligned_free(data);
#endif
#else
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(data);
#endif
}

} // namespace c10
