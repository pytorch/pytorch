#pragma once

#include <torch/csrc/autograd/grad_mode.h>

namespace torch {
using autograd::AutoGradMode;

// A RAII, thread local (!) guard that stops future operations from building
// gradients.
struct TORCH_API NoGradGuard final : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

} // namespace torch
