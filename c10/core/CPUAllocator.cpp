#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/alignment.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/mobile/CPUCachingAllocator.h>
#include <c10/mobile/CPUProfilingAllocator.h>
#include <c10/util/Logging.h>

// TODO: rename flag to C10
C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

namespace c10 {

struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    try {
      data = c10::alloc_cpu(nbytes);
    } catch (c10::Error& e) {
      profiledCPUMemoryReporter().OutOfMemory(nbytes);
      throw e;
    }
    profiledCPUMemoryReporter().New(data, nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    profiledCPUMemoryReporter().Delete(ptr);
    free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
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
  ~DefaultMobileCPUAllocator() override = default;

  static void deleter(void* const pointer) {
    if (C10_UNLIKELY(!pointer)) {
      return;
    }
    // TODO: enable with better TLS support on mobile
    // profiledCPUMemoryReporter().Delete(pointer);
    auto allocator_ptr = GetThreadLocalCachingAllocator();
    auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
    if (allocator_ptr != nullptr) {
      allocator_ptr->free(pointer);
    } else if (profiling_allocator_ptr != nullptr) {
      profiling_allocator_ptr->free(pointer);
    } else {
      c10::free_cpu(pointer);
      // This adds extra cost to freeing memory to the default case when
      // caching allocator is not enabled.
      // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
      CPUCachingAllocator::record_free(pointer);
      auto allocation_planner = GetThreadLocalAllocationPlanner();
      if (allocation_planner != nullptr) {
        allocation_planner->record_free(pointer);
      }
    }
  }

  DataPtr allocate(const size_t nbytes) override {
    if (C10_UNLIKELY(0u == nbytes)) {
      return {
          nullptr,
          nullptr,
          &deleter,
          at::Device(DeviceType::CPU),
      };
    }

    auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void* data;
    auto allocator_ptr = GetThreadLocalCachingAllocator();
    auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
    if (allocator_ptr != nullptr) {
      data = allocator_ptr->allocate(alloc_size);
    } else if (profiling_allocator_ptr != nullptr) {
      data = profiling_allocator_ptr->allocate(alloc_size);
    } else {
      try {
        data = c10::alloc_cpu(alloc_size);
      } catch (c10::Error& e) {
        profiledCPUMemoryReporter().OutOfMemory(alloc_size);
        throw e;
      }
      auto allocation_planner = GetThreadLocalAllocationPlanner();
      if (allocation_planner != nullptr) {
        allocation_planner->record_allocation(alloc_size, data);
      }
    }
    profiledCPUMemoryReporter().New(data, alloc_size);
    return {
        reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
        data,
        &deleter,
        at::Device(DeviceType::CPU),
    };
  }

  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }

  bool is_simple_data_ptr(const c10::DataPtr& data_ptr) const final {
    return reinterpret_cast<const uint8_t*>(data_ptr.get()) ==
        reinterpret_cast<const uint8_t*>(data_ptr.get_context()) +
        PreGuardBytes;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-non-const-global-variables)
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
  if (nbytes == 0) {
    return;
  }
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);
    size_table_[ptr] = nbytes;
    allocated_ += nbytes;
    allocated = allocated_;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 alloc " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    reportMemoryUsageToProfiler(
        ptr,
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

void ProfiledCPUMemoryReporter::Delete(void* ptr) {
  size_t nbytes = 0;
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = size_table_.find(ptr);
    if (it != size_table_.end()) {
      allocated_ -= it->second;
      allocated = allocated_;
      nbytes = it->second;
      size_table_.erase(it);
    } else {
      // C10_LOG_EVERY_MS might log every time in some builds,
      // using a simple counter to avoid spammy logs
      if (log_cnt_++ % 1000 == 0) {
        LOG(WARNING) << "Memory block of unknown size was allocated before "
                     << "the profiling started, profiler results will not "
                     << "include the deallocation event";
      }
    }
  }
  if (nbytes == 0) {
    return;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 deleted " << nbytes << " bytes, total alloc " << allocated
              << " bytes.";
  }
  if (profile_memory) {
    reportMemoryUsageToProfiler(
        ptr,
        -static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

void ProfiledCPUMemoryReporter::OutOfMemory(size_t nbytes) {
  auto profile_memory = memoryProfilingEnabled();
  size_t allocated = 0;
  if (FLAGS_caffe2_report_cpu_memory_usage || profile_memory) {
    std::lock_guard<std::mutex> guard(mutex_);

    allocated = allocated_;
  }
  if (nbytes == 0) {
    return;
  }
  if (FLAGS_caffe2_report_cpu_memory_usage) {
    LOG(INFO) << "C10 Out of Memory. Trying to allocate " << nbytes
              << " bytes, total alloc " << allocated << " bytes.";
  }
  if (profile_memory) {
    reportOutOfMemoryToProfiler(
        static_cast<int64_t>(nbytes),
        allocated,
        0,
        c10::Device(c10::DeviceType::CPU));
  }
}

C10_API at::Allocator* cpu_caching_alloc = nullptr;
C10_API uint8_t cpu_caching_alloc_priority = 0;

void SetCPUCachingAllocator(Allocator* alloc, uint8_t priority) {
  if (priority >= cpu_caching_alloc_priority) {
    cpu_caching_alloc = alloc;
    cpu_caching_alloc_priority = priority;
  }
}

Allocator* GetCPUCachingAllocator() {
  if (cpu_caching_alloc == nullptr) {
    VLOG(1)
        << "There is not caching allocator registered for CPU, use the default allocator instead.";
    return GetAllocator(DeviceType::CPU);
  }
  return cpu_caching_alloc;
}

} // namespace c10
