#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>

#include <ATen/detail/MTIAHooksInterface.h>
#include <optional>

namespace at::accelerator {

// Note [Accelerator Concept]
// This file defines the top level Accelerator concept for PyTorch.
// A device is an accelerator per the definition here if:
// - It is mutually exclusive with all other accelerators
// - It performs asynchronous compute via a Stream/Event system
// - It provides a set of common APIs as defined by AcceleratorHooksInterface
//
// As of today, accelerator devices are (in no particular order):
// CUDA, MTIA, XPU, HIP, MPS, PrivateUse1

// Ensures that only one accelerator is available (at
// compile time if possible) and return it.
// When checked is true, the returned optional always has a value.
TORCH_API std::optional<c10::DeviceType> getAccelerator(bool checked = false);

// Check if the given device type is an accelerator.
TORCH_API bool isAccelerator(c10::DeviceType device_type);

// Check if the given device type is an accelerator, not the excluded ones.
template <
    typename... T,
    typename = std::enable_if_t<(std::is_same_v<T, c10::DeviceType> && ...)>>
inline bool isAcceleratorExcluded(
    c10::DeviceType device_type,
    c10::DeviceType first_excluded,
    T... rest_excluded) {
  if constexpr (sizeof...(rest_excluded) > 0) {
    return device_type != first_excluded &&
        isAcceleratorExcluded(device_type, rest_excluded...);
  } else {
    return device_type != first_excluded && isAccelerator(device_type);
  }
}

// Return the number of the device available. Note that this is *REQUIRED* to
// not raise any exception.
TORCH_API c10::DeviceIndex deviceCount();

// Set the current device index to the given device index.
TORCH_API void setDeviceIndex(c10::DeviceIndex device_index);

// Get the current device index.
TORCH_API c10::DeviceIndex getDeviceIndex();

// Set the current stream to a given stream. Note that this API doesn't change
// the current device index.
TORCH_API void setCurrentStream(c10::Stream stream);

// Get the current stream of the given device index.
TORCH_API c10::Stream getCurrentStream(c10::DeviceIndex device_index);

// Wait (by blocking the calling thread) until all the work previously enqueued
// on the given device index has been completed.
TORCH_API void synchronizeDevice(c10::DeviceIndex device_index);

// Set the current device index to the given device_index and return the
// original device index that was active before the change.
TORCH_API c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index);

// Set the current device index to the given device_index. Avoid creating a new
// context if the context for device_index is not initialized. Return the
// original device index that was active before the change.
TORCH_API c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index);

} // namespace at::accelerator

namespace at {
// Keep BC only
using at::accelerator::getAccelerator;
using at::accelerator::isAccelerator;
} // namespace at
