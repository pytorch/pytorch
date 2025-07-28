#include <c10/core/Allocator.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());

Allocator* get() {
  return &allocator;
}

void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream) {
  HIPCachingAllocator::recordStream(ptr, stream.hip_stream());
}

// Register this HIP allocator as CUDA allocator to enable access through both
// c10::GetAllocator(kCUDA) and c10::getDeviceAllocator(kCUDA) APIs
REGISTER_ALLOCATOR(kCUDA, &allocator)

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
