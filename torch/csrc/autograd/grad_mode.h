#pragma once

#include <ATen/core/grad_mode.h>
#include <torch/csrc/Export.h>

namespace torch {
namespace autograd {

using GradMode = at::GradMode;
using AutoGradMode = at::AutoGradMode;

} // namespace autograd
} // namespace torch
