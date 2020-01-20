#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>

namespace torch {
namespace cuda {
/// Returns the number of CUDA devices available.
size_t TORCH_API device_count();

/// Returns the index of the current CUDA device or -1 if none are available.
int TORCH_API current_device();

/// Sets the index of the current CUDA device.
void TORCH_API set_device(int);

/// Returns true if at least one CUDA device is available.
bool TORCH_API is_available();

/// Returns true if CUDA is available, and CuDNN is available.
bool TORCH_API cudnn_is_available();
} // namespace cuda
} // namespace torch
