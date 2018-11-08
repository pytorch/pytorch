#pragma once

// "Convert" tensor a different type and / or device

#include <ATen/ATen.h>

#include <cstddef>

namespace torch { namespace utils {

// Returns a tensor with the same data as `self` and the specified type and
// device. Returns `self` unmodified if neither the type nor device change;
// otherwise a copy is made.
//
// The `device` argument is only relevant if `type` is a CUDA type. There are
// a few special cases for device:
//
//  - if device is -1 then the returned tensor will be on the current device
//  - if device is nullopt then the returned tensor will be on the same device
//    as `self` if possible; otherwise it will be on the current device.
//
// If `non_blocking` is true, then the copy may be performed asynchronously
// w.r.t the host if `self` is a CPU tensor in pinned memory and `type` is a
// CUDA type. Note that copies between CUDA devices are always asynchronous
// w.r.t the host.
at::Tensor dispatch_type_conversion(
    const at::Tensor& self,
    const at::Type& type,
    c10::optional<int32_t> device_index = c10::nullopt,
    bool non_blocking = false);
}} // namespace torch::utils
