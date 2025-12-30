#pragma once

#include <c10/core/Stream.h>

namespace at::accelerator {

// Set the current stream to a given stream. Note that this API doesn't change
// the current device index.
TORCH_API void setCurrentStream(c10::Stream stream);

// Get the current stream of the given device index.
TORCH_API c10::Stream getCurrentStream(c10::DeviceIndex device_index);

} // namespace at::accelerator
