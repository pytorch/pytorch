#pragma once

#include <cstddef>

namespace torch {
namespace cuda {
/// Returns the number of CUDA devices available.
size_t device_count();

/// Returns true if at least one CUDA device is available.
bool is_available();

/// Returns true if CUDA is available, and CuDNN is available.
bool cudnn_is_available();
} // namespace cuda
} // namespace torch
