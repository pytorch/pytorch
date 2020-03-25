#pragma once

#include <ATen/Parallel.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/api/include/torch/types.h>
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

// Returns true if both t1, t2 are undefined or both are defined and equal
inline bool equal_if_defined(Tensor t1, Tensor t2) {
  return ((!t1.defined() && !t2.defined()) || (t1.defined() && t2.defined() && torch::equal(t1, t2)));
}

} // namespace torch
