#pragma once

#include <ATen/Allocator.h>

namespace at { namespace cuda {

at::Allocator* getPinnedMemoryAllocator();

}} // namespace at::cuda
