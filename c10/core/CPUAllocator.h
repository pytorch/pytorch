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

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
C10_API void memset_junk(void* data, size_t num);

C10_API void* alloc_cpu(size_t nbytes, size_t alignment = gAlignment);
C10_API void free_cpu(void* data);

// Get the CPU Allocator.
C10_API at::Allocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetCPUAllocator(at::Allocator* alloc);

// Get the Default CPU Allocator
C10_API at::Allocator* GetDefaultCPUAllocator();

} // namespace c10
