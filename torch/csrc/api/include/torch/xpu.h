#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

namespace torch::xpu {

/// Returns the number of XPU devices available.
size_t TORCH_API device_count();

/// Returns true if at least one XPU device is available.
bool TORCH_API is_available();

} // namespace torch::xpu
