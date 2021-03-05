#pragma once

#include <c10/core/inference_mode.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

using InferenceMode = c10::InferenceMode;
using AutoInferenceMode= c10::AutoInferenceMode;

}}
