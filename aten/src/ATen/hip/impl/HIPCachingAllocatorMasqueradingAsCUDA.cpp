#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

c10::cuda::CUDACachingAllocator::CUDAAllocator* get() {
  return c10::cuda::CUDACachingAllocator::get();
}

void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream) {
  c10::cuda::CUDACachingAllocator::recordStream(ptr, stream);
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
