#pragma once

#include <c10/core/Allocator.h>

namespace c10 {

// Get the Default Mobile CPU Allocator
C10_API at::Allocator* GetDefaultMobileCPUAllocator();

} // namespace c10
