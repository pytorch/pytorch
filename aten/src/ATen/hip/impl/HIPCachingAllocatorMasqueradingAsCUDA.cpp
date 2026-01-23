#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

HIPCachingAllocator::HIPAllocator* get() {
  static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());
  return &allocator;
}

void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream) {
  HIPCachingAllocator::recordStream(ptr, stream.hip_stream());
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
