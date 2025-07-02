#include "OpenRegDeviceAllocator.h"

static OpenRegDeviceAllocator global_openreg_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);
