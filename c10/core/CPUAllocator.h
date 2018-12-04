#pragma once

#include <cstring>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <c10/util/numa.h>

// TODO: rename to c10
C10_DECLARE_bool(caffe2_report_cpu_memory_usage);
C10_DECLARE_bool(caffe2_cpu_allocator_do_zero_fill);
C10_DECLARE_bool(caffe2_cpu_allocator_do_junk_fill);

namespace c10 {

// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
C10_API void NoDelete(void*);

// TODO: delete
// A virtual allocator class to do memory allocation and deallocation.
struct C10_API CPUAllocator {
  CPUAllocator() {}
  virtual ~CPUAllocator() noexcept {}
  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) = 0;
  virtual MemoryDeleter GetDeleter() = 0;
};

// A virtual struct that is used to report C10's memory allocation and
// deallocation status
class C10_API MemoryAllocationReporter {
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
C10_API void memset_junk(void* data, size_t num);

void* alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }

  void* data;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
#else
  CAFFE_ENFORCE_EQ(posix_memalign(&data, gAlignment, nbytes), 0);
#endif

  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate %dGB. Buy new RAM!",
      nbytes / 1073741824);
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

struct C10_API DefaultCPUAllocator final : at::Allocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc(nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      getMemoryAllocationReporter().New(data, nbytes);
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
    getMemoryAllocationReporter().Delete(ptr);
    Delete(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      return &ReportAndDelete;
    }
    return &Delete;
  }

 protected:
  static MemoryAllocationReporter& getMemoryAllocationReporter();

};

// Get the CPU Alloctor.
C10_API at::Allocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetCPUAllocator(at::Allocator* alloc);

} // namespace c10
