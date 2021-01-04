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

#ifdef C10_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
constexpr size_t gAlignment = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;
#endif

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
C10_API void NoDelete(void*);

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
C10_API void memset_junk(void* data, size_t num);

C10_API void* alloc_cpu(size_t nbytes);
C10_API void free_cpu(void* data);

// A simple struct that is used to report C10's memory allocation and
// deallocation status to the profiler
class C10_API ProfiledCPUMemoryReporter {
 public:
  ProfiledCPUMemoryReporter() {}
  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_ = 0;
};

C10_API ProfiledCPUMemoryReporter& profiledCPUMemoryReporter();

// Get the CPU Allocator.
C10_API at::Allocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetCPUAllocator(at::Allocator* alloc, uint8_t priority = 0);

// Get the Default CPU Allocator
C10_API at::Allocator* GetDefaultCPUAllocator();

// Get the Default Mobile CPU Allocator
C10_API at::Allocator* GetDefaultMobileCPUAllocator();

// The CPUCachingAllocator is experimental and might disappear in the future.
// The only place that uses it is in StaticRuntime.
// Set the CPU Caching Allocator
C10_API void SetCPUCachingAllocator(Allocator* alloc, uint8_t priority = 0);
// Get the CPU Caching Allocator
C10_API Allocator* GetCPUCachingAllocator();

} // namespace c10
