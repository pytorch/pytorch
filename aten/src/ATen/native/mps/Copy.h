//  Copyright © 2022 Apple Inc.

#pragma once
#include <ATen/core/Tensor.h>

namespace at {
namespace native {
namespace mps {

at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking);
void copy_blit_mps(void* dst, const void* src, size_t size);

} // namespace mps
} // namespace native
} // namespace at
