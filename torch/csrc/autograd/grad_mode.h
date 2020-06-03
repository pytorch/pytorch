#pragma once

#include <ATen/core/grad_mode.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

using GradMode = at::GradMode;
using FwGradMode = at::FwGradMode;
using AutoGradMode = at::AutoGradMode;

}}
