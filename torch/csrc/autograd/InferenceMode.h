#pragma once

#include <c10/core/InferenceMode.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

using InferenceMode = c10::InferenceMode;

}}
