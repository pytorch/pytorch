#pragma once

#include <ATen/core/amp_mode.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

using AmpMode = at::AmpMode;

}}
