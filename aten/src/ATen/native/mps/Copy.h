//  Copyright © 2022 Apple Inc.

#pragma once
#include <atomic>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/mps/MPSDevice.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at {
namespace native {
namespace mps {

at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking);
void copy_blit_mps(void* dst, const void* src, size_t size);

} // namespace mps
} // namespace native
} // namespace at
