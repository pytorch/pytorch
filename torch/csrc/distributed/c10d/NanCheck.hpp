#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>

namespace c10d {

// Check for NaNs in a tensor. If any are found, throw an error.
// Dispatches to device-specific implementations via the c10d::check_for_nan op.
TORCH_API void checkForNan(const at::Tensor& tensor);

} // namespace c10d
