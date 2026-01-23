#include <c10/core/CachingDeviceAllocator.h>

namespace c10 {

// Ensures proper DLL export of this pure virtual base class on Windows,
// since it's mainly used in other DLLs outside c10.dll.
DeviceAllocator::DeviceAllocator() = default;
DeviceAllocator::~DeviceAllocator() = default;

} // namespace c10
