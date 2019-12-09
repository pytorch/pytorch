#pragma once

#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

Allocator* get();
C10_HIP_API void recordStreamMasqueradingAsCUDA(void *ptr, HIPStreamMasqueradingAsCUDA stream);

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
