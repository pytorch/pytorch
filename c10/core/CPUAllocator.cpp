#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>

// TODO: rename flags to C10
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU");

namespace c10 {

void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  int32_t int64_count = num / sizeof(kJunkPattern64);
  int32_t remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (int i = 0; i < int64_count; i++) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

void* alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
    ((ptrdiff_t)nbytes) >= 0,
    "alloc_cpu() seems to have been called with negative number: ", nbytes);

  void* data;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
#else
  int err = posix_memalign(&data, gAlignment, nbytes);
  if (err != 0) {
    CAFFE_THROW(
        "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
        nbytes,
        " bytes. Error code ",
        err,
        " (",
        strerror(err),
        ")");
  }
#endif

  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes. Buy new RAM!");

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
  _aligned_free(data);
#else
  free(data);
#endif
}

struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc_cpu(nbytes);
    profiledCPUMemoryReporter().New(data, nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage && nbytes > 0) {
      return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
    }
    return {data, data, &free_cpu, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      return &ReportAndDelete;
    }
    return &free_cpu;
  }
};

ProfiledCPUMemoryReporter& profiledCPUMemoryReporter() {
  static ProfiledCPUMemoryReporter reporter_;
  return reporter_;
}

// QNNPACK AND XNNPACK may out-of-bound access the input and / or output
// tensors. This is by-design, and chosen to make the implementation of
// micro-kernels both simpler and faster as a result of not having to
// individually handle the corner cases where the number of processed elements
// is not a multiple of SIMD register width.  This behavior will trigger ASAN
// though, and may result in a segfault if the accessed memory location just so
// happens to fall on a page the current process has no read access to.  Here we
// define a custom allocator that allocates the extra storage required to keep
// this behavior safe.  This allocator could have been restricted to QNNPACK and
// XNNPACK only, but that would have negative performance ramifications, as
// input tensors must now be reallocated, and copied over, if the tensor is not
// allocated with this allocator to begin with.  Making this allocator the
// default on mobile builds minimizes the probability of unnecessary
// reallocations and copies, and also enables acceleration of operations where
// the output tensor is allocated outside of the function doing the
// implementation, wherein the implementation cannot simply re-allocate the
// output with the guarding allocator.
//
// PreGuardBytes: Number of guard bytes to allocate before the allocation.
// PostGuardBytes: Number of guard bytes to allocate after the allocation.

template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class DefaultMobileCPUAllocator final : public at::Allocator {
 public:
  DefaultMobileCPUAllocator() = default;
  virtual ~DefaultMobileCPUAllocator() override = default;

  static void deleter(void* const pointer) {
    if (C10_UNLIKELY(!pointer)) {
      return;
    }

    profiledCPUMemoryReporter().Delete(pointer);
    c10::free_cpu(pointer);
  }

  virtual DataPtr allocate(const size_t nbytes) const override {
    if (C10_UNLIKELY(0u == nbytes)) {
      return {
          nullptr,
          nullptr,
          &deleter,
          at::Device(DeviceType::CPU),
      };
    }

    auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
    void* const data = c10::alloc_cpu(alloc_size);

    profiledCPUMemoryReporter().New(data, alloc_size);

    return {
        reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
        data,
        &deleter,
        at::Device(DeviceType::CPU),
    };
  }

  virtual DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
};

void NoDelete(void*) {}

at::Allocator* GetCPUAllocator() {
  return GetAllocator(DeviceType::CPU);
}

void SetCPUAllocator(at::Allocator* alloc, uint8_t priority) {
  SetAllocator(DeviceType::CPU, alloc, priority);
}

// The Mobile CPU allocator must always be present even on non-mobile builds
// because QNNPACK and XNNPACK are not mobile specific.
//
// Pre-guard: 8 bytes for QNNPACK, but set to gAlignment to ensure SIMD
//            alignment, not on the allocated memory, but memory location
//            returned to the user.
// Post-guard: 16 bytes for XNNPACK.

static DefaultMobileCPUAllocator<gAlignment, 16u> g_mobile_cpu_allocator;

at::Allocator* GetDefaultMobileCPUAllocator() {
  return &g_mobile_cpu_allocator;
}

#ifdef C10_MOBILE

at::Allocator* GetDefaultCPUAllocator() {
  return GetDefaultMobileCPUAllocator();
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_mobile_cpu_allocator);

#else

// Global default CPU Allocator
static DefaultCPUAllocator g_cpu_alloc;

at::Allocator* GetDefaultCPUAllocator() {
  return &g_cpu_alloc;
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_alloc);

#endif /* C10_Mobile */

void ProfiledCPUMemoryReporter::New(void* ptr, size_t nbytes) {
  if (memoryProfilingEnabled()) {
    std::lock_guard<std::mutex> guard(mutex_);
    size_table_[ptr] = nbytes;
    reportMemoryUsageToProfiler(c10::Device(c10::DeviceType::CPU), nbytes);
  }
}

void ProfiledCPUMemoryReporter::Delete(void* ptr) {
  if (memoryProfilingEnabled()) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = size_table_.find(ptr);
    CHECK(it != size_table_.end());
    reportMemoryUsageToProfiler(c10::Device(c10::DeviceType::CPU), -it->second);
    size_table_.erase(it);
  }
}

} // namespace c10
