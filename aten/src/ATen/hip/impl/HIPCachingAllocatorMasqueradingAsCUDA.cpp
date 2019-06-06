#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

Allocator* get() {
  static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());
  return &allocator;
}

void recordStreamMasqueradingAsCUDA(
    void *ptr, HIPStreamMasqueradingAsCUDA stream, bool suppressError) {
  HIPCachingAllocator::recordStream(ptr, stream.hip_stream(), suppressError);
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
