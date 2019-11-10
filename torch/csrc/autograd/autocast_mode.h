#pragma once

#include <ATen/core/autocast/autocast_mode.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

// I'm not sure why this header is even necessary (couldn't I just say
// at::AutocastMode directly in init.cpp??) but I'm imitating grad_mode.h,
// which presumably had good intentions.

namespace torch { namespace autograd {

using AutocastMode = at::autocast::AutocastMode;

}}
