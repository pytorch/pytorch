#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceCapability.h>

namespace at::accelerator {

// Return the number of the device available. Note that this is *REQUIRED* to
// not raise any exception.
TORCH_API c10::DeviceIndex deviceCount();

// Set the current device index to the given device index.
TORCH_API void setDeviceIndex(c10::DeviceIndex device_index);

// Get the current device index.
TORCH_API c10::DeviceIndex getDeviceIndex();

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

// Get the device capability of the given device index.
TORCH_API c10::DeviceCapability getDeviceCapability(
    c10::DeviceIndex device_index);

} // namespace at::accelerator
