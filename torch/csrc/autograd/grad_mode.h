#pragma once

#include <ATen/core/grad_mode.h>
#include <torch/csrc/Export.h>

namespace torch::autograd {

using GradMode = at::GradMode;
using AutoGradMode = at::AutoGradMode;

} // namespace torch::autograd
