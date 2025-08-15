#include "OpenRegDeviceAllocator.h"

namespace c10::openreg {

static OpenRegDeviceAllocator global_openreg_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);

} // namespace c10::openreg
