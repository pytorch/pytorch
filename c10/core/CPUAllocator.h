#pragma once

#include <cstring>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/core/alignment.h> // legacy, update dependents to include this directly
#include <c10/util/Logging.h>

// TODO: rename to c10
C10_DECLARE_bool(caffe2_report_cpu_memory_usage);

namespace c10 {

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
C10_API void NoDelete(void*);

// A simple struct that is used to report C10's memory allocation,
// deallocation status and out-of-memory events to the profiler
class C10_API ProfiledCPUMemoryReporter {
 public:
  ProfiledCPUMemoryReporter() {}
  void New(void* ptr, size_t nbytes);
  void OutOfMemory(size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_ = 0;
  size_t log_cnt_ = 0;
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
