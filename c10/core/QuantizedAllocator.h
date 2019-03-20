#pragma once

#include <cstring>
#include <unordered_map>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Logging.h>

namespace c10 {

C10_API void* alloc_quantized(size_t nbytes);
C10_API void free_quantized(void* data);

// Get the Quantized Alloctor.
C10_API at::Allocator* GetQuantizedAllocator();
// Sets the Quantized allocator to the given allocator: the caller gives away the
// ownership of the pointer.
C10_API void SetQuantizedAllocator(at::Allocator* alloc);

// Get the Default Quantized Allocator
C10_API at::Allocator* GetDefaultQuantizedAllocator();

} // namespace c10
