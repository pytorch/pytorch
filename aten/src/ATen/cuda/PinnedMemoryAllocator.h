#pragma once

#include <c10/core/Allocator.h>

namespace at { namespace cuda {

at::Allocator* getPinnedMemoryAllocator();

}} // namespace at::cuda
