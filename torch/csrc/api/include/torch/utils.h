#pragma once

#include <torch/csrc/autograd/grad_mode.h>

#include <cstdint>

namespace torch {
using autograd::AutoGradMode;

// A RAII, thread local (!) guard that stops future operations from building
// gradients.
struct NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

/// Sets the global random seed for all newly created CPU and CUDA tensors.
void manual_seed(uint64_t seed);
} // namespace torch
