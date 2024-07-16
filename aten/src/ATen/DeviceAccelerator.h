#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>

#include <ATen/detail/MTIAHooksInterface.h>
#include <optional>

// This file defines the top level Accelerator concept for PyTorch.
// A device is an accelerator per the definition here if:
// - It is mutually exclusive with all other accelerators
// - It performs asynchronous compute via a Stream/Event system
// - It provides a set of common APIs as defined by AcceleratorHooksInterface
//
// As of today, accelerator devices are (in no particular order):
// CUDA, MTIA, XPU, PrivateUse1
// We want to add once all the proper APIs are supported and tested:
// HIP, MPS

namespace at {

// Ensures that only one accelerator is available (at
// compile time if possible) and return it.
// When checked is true, the returned optional always has a value.
TORCH_API std::optional<c10::DeviceType> getAccelerator(bool checked = false);

} // namespace at
