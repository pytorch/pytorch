#pragma once

#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

Allocator* get();

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
