#include "c10/core/HugePagesAllocator.h"
#include "caffe2/core/context.h"

#include <jemalloc/jemalloc.h>
#include <sys/mman.h>
#include <iostream>

template <typename T>
c10::optional<T> jemallctl_get( // NOLINT(readability-identifier-naming)
    const std::string& ctlName) {
  T rv;
  size_t sz = sizeof(T);
  if (auto ret = jemallctl(ctlName.c_str(), &rv, &sz, nullptr, 0)) {
    std::cerr << "jemallctl: unable to read " << ctlName << ": "
              << strerror(ret);
    return {};
  }
  return {rv};
}

template <typename T>
bool jemallctl_set( // NOLINT(readability-identifier-naming)
    const std::string& ctlName,
    T value) {
  if (auto ret =
          jemallctl(ctlName.c_str(), nullptr, nullptr, &value, sizeof(T))) {
    std::cerr << "jemallctl: unable to set " << ctlName << ": "
              << strerror(ret);
    return false;
  }
  return true;
}

int64_t time_since_epoch() {
  auto t = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t.time_since_epoch())
      .count();
}

std::string get_stats() {
  // Update the statistics cached by mallctl.
  uint64_t epoch = 1;
  size_t sz;
  sz = sizeof(epoch);
  jemallctl("epoch", &epoch, &sz, &epoch, sz);

  // Get basic allocation statistics.  Take care to check for
  // errors, since --enable-stats must have been specified at
  // build time for these statistics to be available.
  size_t allocated, active, metadata, resident, mapped, retained;
  std::ostringstream out;
  sz = sizeof(size_t);
  if (jemallctl("stats.allocated", &allocated, &sz, NULL, 0) == 0 &&
      jemallctl("stats.active", &active, &sz, NULL, 0) == 0 &&
      jemallctl("stats.metadata", &metadata, &sz, NULL, 0) == 0 &&
      jemallctl("stats.resident", &resident, &sz, NULL, 0) == 0 &&
      jemallctl("stats.mapped", &mapped, &sz, NULL, 0) == 0 &&
      jemallctl("stats.retained", &retained, &sz, NULL, 0) == 0) {
    out << allocated << "," << active << "," << metadata << "," << resident
        << "," << mapped << "," << retained;
  }
  return out.str();
}

// OversizeArena gflags
C10_DEFINE_int64(
    caffe2_dirty_decay_ms,
    60000,
    "Duration of the dirty decay phase in ms.");

C10_DEFINE_int64(
    caffe2_muzzy_decay_ms,
    60000,
    "Duration of the muzzy decay phase in ms.");

// Threshold: 128 * 1024 * 1024 = 134,217,728
C10_DEFINE_int64(
    caffe2_jemalloc_oversize_threshold,
    8LL << 20,
    "The threshold in bytes of which requests are considered oversize. Allocation "
    "requests with sizes >= caffe2_jemalloc_oversize_threshold are fulfilled from a "
    "dedicated arena (automatically managed, however not within narenas), in order "
    "to reduce fragmentation by not mixing huge allocations with small ones. "
    "Allocation requests with sizes < caffe2_jemalloc_oversize_threshold will be "
    "cached and they will follow the two-phase decay curve. In addition, the decay "
    "API guarantees on the extents greater than the specified threshold may be "
    "overridden. The default threshold is 8MB.");

// HugePagesArena gflag
C10_DEFINE_int64(
    caffe2_huge_pages_threshold,
    2LL << 20,
    "Allocate 2MB pages for allocations more tham this. To disable set to -1.");

/**
 * An allocator which uses Jemalloc to create dedicated arenas to allocate
 * memory from.
 *
 * If enabled, we put memory blocks in an arena with customizable decay
 * settings. This is done by setting arena.<idx>.muzzy_decay_ms to -1 (turning
 * off decay) or large a none-zero number.
 *
 * Memory blocks larger than arena.<idx>.oversize_threshold do not follow the
 * two-phase decay. Set it to a large number (e.g. 128 MB) so that the larger
 * allocations (smaller than arena.<idx>.oversize_threshold) do get released
 * following the two-phase decay curve.
 */

namespace c10 {

struct OversizeArena {
 public:
  static OversizeArena& instance() {
    static OversizeArena instance;
    return instance;
  }

  c10::optional<int> oversize_flags_;
  size_t tcache_max_;

  bool isEnabled() const {
    return oversize_flags_.has_value();
  }

  int getOversizeArenaFlags() const {
    return *oversize_flags_;
  }

  size_t getTCacheMax() const {
    return tcache_max_;
  }

  static void* alloc(size_t size) {
    if (!instance().isEnabled()) {
      std::cout << "Not using OversizeArena";
      return nullptr;
    }

    void* result = jemallocx(size, instance().getOversizeArenaFlags());
    if (!result) {
      std::cerr << "Can't allocate in OversizeArena for " << size
                << " bytes: " << strerror(errno);
      return nullptr;
    }
//    if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
//      std::cout << "alloc," << time_since_epoch() << "," << result << ","
//                << get_stats() << std::endl;
//    }
    return result;
  }

  static void dealloc(void* ptr) {
    if (!instance().isEnabled()) {
      jefree(ptr);
//      if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
//        std::cout << "free," << time_since_epoch() << "," << ptr << ","
//                  << get_stats() << std::endl;
//      }
      return;
    }
    if (nullptr == ptr) {
      return;
    }
    // dallocx() seems not dependent on arena index or alignment, so we use a
    // minimal flag for a common dealloc() function for both arenas.
    jedallocx(ptr, MALLOCX_TCACHE_NONE);
//    if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
//      std::cout << "free," << time_since_epoch() << "," << ptr << ","
//                << get_stats() << std::endl;
//    }
  }

 private:
  OversizeArena() : tcache_max_(0) {
    oversize_flags_.reset();
    // create another arena for oversize allocations
    auto arenaIndex = jemallctl_get<unsigned int>("arenas.create");
    if (!arenaIndex.has_value()) {
      return;
    }
//    std::cout << "arenaIndex: " << arenaIndex.value() << std::endl;
    // set decay times
    CAFFE_ENFORCE(
        FLAGS_caffe2_dirty_decay_ms >= -1,
        "caffe2_dirty_decay_ms needs to be >= -1, but is ",
        FLAGS_caffe2_dirty_decay_ms);
    auto ctl = "arena." + c10::to_string(*arenaIndex) + ".dirty_decay_ms";
    jemallctl_set(ctl, FLAGS_caffe2_dirty_decay_ms);

    CAFFE_ENFORCE(
        FLAGS_caffe2_muzzy_decay_ms >= -1,
        "caffe2_muzzy_decay_ms needs to be >= -1 but is",
        FLAGS_caffe2_muzzy_decay_ms);
    ctl = "arena." + c10::to_string(*arenaIndex) + ".muzzy_decay_ms";
    jemallctl_set(ctl, FLAGS_caffe2_muzzy_decay_ms);

    // set oversize_threshold
    CAFFE_ENFORCE(
        FLAGS_caffe2_jemalloc_oversize_threshold >= 0,
        "caffe2_jemalloc_oversize_threshold needs to be >= 0, but is ",
        FLAGS_caffe2_jemalloc_oversize_threshold);
    ctl = "arena." + c10::to_string(*arenaIndex) + ".oversize_threshold";
    jemallctl_set(ctl, FLAGS_caffe2_jemalloc_oversize_threshold);

    tcache_max_ = jemallctl_get<size_t>("opt.tcache_max").value();

    // no need to align on 2MB boundary for the oversize arena
    oversize_flags_ = MALLOCX_ARENA(*arenaIndex) | MALLOCX_TCACHE_NONE;
  }
};

/**
 * An allocator which uses Jemalloc to create dedicated arenas to allocate
 * memory from.
 *
 * For 'huge' memory blocks (>=caffe2_huge_pages_threshold), we're going to
 * ask kernel to put memory on 2MB pages if possible.
 *
 * This is done by setting MADV_HUGEPAGE using the `madvise` system call.
 * Jemalloc does not use allocated chunks / extents across different arenas,
 * without `munmap`-ing them first, and the advises are not sticky i.e. they are
 * unset if `munmap` is done.
 */

struct HugePagesArena {
 public:
  static HugePagesArena& instance() {
    static HugePagesArena instance;
    return instance;
  }
  // fields are static as there's only 1 instance anyway
  c10::optional<int> huge_pages_flags_;

  bool isEnabled() {
    return huge_pages_flags_.has_value();
  }

  int getHugePagesArenaFlags() {
    return *huge_pages_flags_;
  }

  static void* alloc(size_t size) {
    if (!instance().isEnabled()) {
      return nullptr;
    }
    void* result = jemallocx(size, instance().getHugePagesArenaFlags());
    if (!result) {
      std::cerr << "Can't allocate huge pages for " << size
                << " bytes: " << strerror(errno);
      return nullptr;
    }
    if (auto ret = madvise(result, size, MADV_HUGEPAGE)) {
      std::cerr << "Can't mark as huge pages for " << size
                << " bytes: " << strerror(ret);
    }
    return result;
  }

  static void dealloc(void* ptr) {
    if (!instance().isEnabled()) {
      jefree(ptr);
      return;
    }
    if (nullptr == ptr) {
      return;
    }
    // dallocx() seems not dependent on arena index or alignment, so we use a
    // minimal flag for a common dealloc() function for both arenas.
    jedallocx(ptr, MALLOCX_TCACHE_NONE);
  }

 private:
  HugePagesArena() {
    huge_pages_flags_.reset();
    auto arenaIndex = jemallctl_get<unsigned int>("arenas.create");
    if (!arenaIndex.has_value()) {
      return;
    }
    // align on 2MB boundary for the huge page arena
    huge_pages_flags_ =
        MALLOCX_ARENA(*arenaIndex) | MALLOCX_LG_ALIGN(21) | MALLOCX_TCACHE_NONE;
  }
};

void* je_alloc_cpu(size_t nbytes) {
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
  int err = jeposix_memalign(&data, gAlignment, nbytes);
  if (err != 0) {
    CAFFE_THROW(
        "JEMallocCPUAllocator: can't allocate memory: you tried to allocate ",
        nbytes,
        " bytes. Error code ",
        err,
        " (",
        strerror(err),
        ")");
  }

  CAFFE_ENFORCE(
      data,
      "JEMallocCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");

  // move data to a thread's NUMA node
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    memset_junk(data, nbytes);
  }

//  if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
//    std::cout << "alloc," << time_since_epoch() << "," << data << ","
//              << get_stats() << std::endl;
//  }

  return data;
}

void je_free_cpu(void* data) {
  jefree(data);
//  if (const char* env_p = std::getenv("PRINT_JEMALLOC_HEAP")) {
//    std::cout << "free," << time_since_epoch() << "," << data << ","
//              << get_stats() << std::endl;
//  }
}

struct JEMallocCPUAllocator final : at::Allocator {
  JEMallocCPUAllocator() {
//    std::cout << "JEMallocCPUAllocator cons called\n";
  }
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = je_alloc_cpu(nbytes);
    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    je_free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

static JEMallocCPUAllocator g_je_malloc_cpu_alloc;

bool installJEMallocCPUAllocator() {
  // Setting allocator globally with priority = 1, it will not
  // be overrridden unless another CPU allocator is set with a
  // higher priority value.
  c10::SetCPUAllocator(&g_je_malloc_cpu_alloc, 1 /* priority */);
  return true;
}

// install the allocator statically
//static bool registry = installJEMallocCPUAllocator();

struct HugePagesCpuAllocator final : at::Allocator {
  HugePagesCpuAllocator() {
//    std::cout
//        << "HugePagesCpuAllocator cons called; FLAGS_caffe2_huge_pages_threshold: "
//        << FLAGS_caffe2_huge_pages_threshold << std::endl;
    baseAllocator_ = &g_je_malloc_cpu_alloc;
  }
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
    if (FLAGS_caffe2_huge_pages_threshold > 0 &&
        nbytes >= FLAGS_caffe2_huge_pages_threshold) {
      data = HugePagesArena::alloc(nbytes);
//      std::cout << "Allocated " << nbytes << " on huge pages";
    }
    if (!data) {
//      std::cout << "Allocated " << nbytes << " on regular pages";
      return baseAllocator_->allocate(nbytes);
    }
    at::profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(caffe2::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    at::profiledCPUMemoryReporter().Delete(ptr);
    HugePagesArena::dealloc(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

 private:
  at::Allocator* baseAllocator_;
};

// static HugePagesCpuAllocator g_huge_pages_cpu_alloc;

// bool installHugePagesAllocator() {
//   // Setting allocator globally with priority = 1, it will not
//   // be overrridden unless another CPU allocator is set with a
//   // higher priority value.
//   c10::SetCPUAllocator(&g_huge_pages_cpu_alloc, 1 /* priority */);
//   return true;
// }

// install the allocator statically
// static bool registry = installHugePagesAllocator();

int str2int(const string& str) {
  std::stringstream ss(str);
  int num;
  if ((ss >> num).fail()) {
    CAFFE_THROW("couldn't parse num");
  }
  return num;
}

struct OversizeArenaAllocator final : at::Allocator {
  OversizeArenaAllocator() : oversize_arena_(OversizeArena::instance()) {
    //    tcache_max_ = oversize_arena_.getTCacheMax();
    auto thresh = str2int(std::getenv("OVERSIZED_THRESHOLD"));
    if (thresh < 0) {
      oversized_threshold_ = std::numeric_limits<int>::max();
    } else {
      oversized_threshold_ = thresh;
    }
    baseAllocator_ = &g_je_malloc_cpu_alloc;
  }

  at::DataPtr allocate(size_t nbytes) const override {
    if (nbytes <= oversized_threshold_ ||
        !oversize_arena_.isEnabled()) {

      return baseAllocator_->allocate(nbytes);
    }
    void* data = OversizeArena::alloc(nbytes);
    TORCH_CHECK(data != nullptr, "Failed to allocate ", nbytes, "bytes");

    at::profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(caffe2::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    at::profiledCPUMemoryReporter().Delete(ptr);
    OversizeArena::dealloc(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

 private:
  const OversizeArena& oversize_arena_;
  size_t oversized_threshold_;
  at::Allocator* baseAllocator_;
};

static OversizeArenaAllocator g_oversize_cpu_alloc;

C10_API bool installOversizeAllocator() {
  // Setting allocator globally with priority = 2 to override
  // caffe2::AllocationArenaPool. It will not be overrridden unless another CPU
  // allocator is set with a higher priority value.
  c10::SetCPUAllocator(&g_oversize_cpu_alloc, 1 /* priority */);
//  std::cout << "installed oversize allocator" << std::endl;
  return true;
}

// install the allocator statically
static bool registry = installOversizeAllocator();

} // namespace c10