#ifndef CAFFE2_CORE_ALLOCATOR_H_
#define CAFFE2_CORE_ALLOCATOR_H_

#include <cstring>
#include <unordered_map>

#include <ATen/core/Allocator.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/numa.h"

C10_DECLARE_bool(caffe2_report_cpu_memory_usage);
C10_DECLARE_bool(caffe2_cpu_allocator_do_zero_fill);
C10_DECLARE_bool(caffe2_cpu_allocator_do_junk_fill);

namespace caffe2 {

// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gCaffe2Alignment = 64;

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
CAFFE2_API void NoDelete(void*);

// A virtual allocator class to do memory allocation and deallocation.
struct CAFFE2_API CPUAllocator {
  CPUAllocator() {}
  virtual ~CPUAllocator() noexcept {}
  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) = 0;
  virtual MemoryDeleter GetDeleter() = 0;
};

// A virtual struct that is used to report Caffe2's memory allocation and
// deallocation status
class CAFFE2_API MemoryAllocationReporter {
 public:
  MemoryAllocationReporter() : allocated_(0) {}
  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
CAFFE2_API void memset_junk(void* data, size_t num);

struct CAFFE2_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
#ifdef __ANDROID__
    data = memalign(gCaffe2Alignment, nbytes);
#elif defined(_MSC_VER)
    data = _aligned_malloc(nbytes, gCaffe2Alignment);
#else
    CAFFE_ENFORCE_EQ(posix_memalign(&data, gCaffe2Alignment, nbytes), 0);
#endif
    CAFFE_ENFORCE(data);
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
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      reporter_.New(data, nbytes);
      return {data, data, &ReportAndDelete, at::Device(at::DeviceType::CPU)};
    }
    return {data, data, &Delete, at::Device(at::DeviceType::CPU)};
  }

#ifdef _MSC_VER
  static void Delete(void* data) {
    _aligned_free(data);
  }
#else
  static void Delete(void* data) {
    free(data);
  }
#endif

  static void ReportAndDelete(void* ptr) {
    reporter_.Delete(ptr);
    Delete(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      return &ReportAndDelete;
    }
    return &Delete;
  }

 protected:
  static MemoryAllocationReporter reporter_;
};

// Get the CPU Alloctor.
CAFFE2_API at::Allocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
CAFFE2_API void SetCPUAllocator(at::Allocator* alloc);

} // namespace caffe2

#endif // CAFFE2_CORE_ALLOCATOR_H_
