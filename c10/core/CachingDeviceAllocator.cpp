#include <c10/core/CachingDeviceAllocator.h>

namespace c10 {

// DeviceAllocator::DeviceAllocator() = default;
// DeviceAllocator::~DeviceAllocator() = default;

DeviceAllocator* GetDeviceAllocator(const DeviceType& t) {
  auto* allocator = c10::GetAllocator(t);
  auto* device_allocator = dynamic_cast<DeviceAllocator*>(allocator);
  TORCH_INTERNAL_ASSERT(
      device_allocator, "Allocator for ", t, " is not a DeviceAllocator.");
  return device_allocator;
}

} // namespace c10
