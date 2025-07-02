#include "OpenRegHostAllocator.h"

OpenRegHostAllocator caching_host_allocator;

REGISTER_HOST_ALLOCATOR(at::kPrivateUse1, &caching_host_allocator);
