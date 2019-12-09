#pragma once

#include <ATen/Parallel.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <cstdint>

namespace torch {

using NoGradGuard = at::NoGradGuard;

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;

// Called during new thread initialization
using at::init_num_threads;

// Returns the number of threads used in parallel region.
using at::get_num_threads;

// Sets the number of threads to be used in parallel region.
using at::set_num_threads;

// Returns the number of threads used for inter-op parallelism.
using at::get_num_interop_threads;

// Sets the number of threads to be used for inter-op parallelism.
using at::set_num_interop_threads;

} // namespace torch
