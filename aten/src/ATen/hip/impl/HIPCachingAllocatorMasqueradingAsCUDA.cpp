#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

namespace c10 { namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

Allocator* get() {
  static HIPAllocatorMasqueradingAsCUDA allocator(HIPCachingAllocator::get());
  return &allocator;
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
}} // namespace c10::hip
