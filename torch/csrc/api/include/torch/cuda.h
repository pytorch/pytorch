#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <cstdint>

namespace torch::cuda {

/// Returns the number of CUDA devices available.
c10::DeviceIndex TORCH_API device_count();

/// Returns true if at least one CUDA device is available.
bool TORCH_API is_available();

/// Returns true if CUDA is available, and CuDNN is available.
bool TORCH_API cudnn_is_available();

/// Sets the seed for the current GPU.
void TORCH_API manual_seed(uint64_t seed);

/// Sets the seed for all available GPUs.
void TORCH_API manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a CUDA device to complete.
void TORCH_API synchronize(int64_t device_index = -1);

} // namespace torch::cuda
