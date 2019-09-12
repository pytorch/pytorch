#pragma once
#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>

 #include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

float clip_grad_norm_(
    std::vector<Tensor>& parameters,
    float max_norm,
    float norm_type = 2.0);

} // namespace nn
} // namespace torch
