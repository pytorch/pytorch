#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

namespace torch {
namespace mps {

/// Returns true if MPS device is available.
bool TORCH_API is_available();

/// Sets the RNG seed for the MPS device.
void TORCH_API manual_seed(uint64_t seed);

/// Waits for all streams on a MPS device to complete.
void TORCH_API synchronize();

} // namespace mps
} // namespace torch
