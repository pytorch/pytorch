#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/Context.h>
#include <torch/types.h>

namespace torch {

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;

/// Returns the random number generator state as a ByteTensor.
Tensor TORCH_API get_rng_state();

/// Sets the random number generator state.
void TORCH_API set_rng_state(const Tensor & new_state);

namespace cuda {

/// Returns the random number generator state of the specified GPU as a ByteTensor.
Tensor TORCH_API get_rng_state(torch::DeviceIndex device_index);

/// Returns the random number generator state of the specified GPU as a ByteTensor.
Tensor TORCH_API get_rng_state(torch::Device device = torch::Device(torch::kCUDA));

/// Sets the random number generator state of the specified GPU.
void TORCH_API set_rng_state(const Tensor & new_state, torch::DeviceIndex device_index);

/// Sets the random number generator state of the specified GPU.
void TORCH_API set_rng_state(const Tensor & new_state, torch::Device device = torch::Device(torch::kCUDA));

}

}  // namespace torch
