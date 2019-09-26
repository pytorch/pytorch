#pragma once
#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {
namespace utils {

float clip_grad_norm_(
    std::vector<Tensor>& parameters,
    float max_norm,
    float norm_type = 2.0);

float clip_grad_norm_(
    Tensor& parameters,
    float max_norm,
    float norm_type = 2.0);

} // namespace utils
} // namespace nn
} // namespace torch
