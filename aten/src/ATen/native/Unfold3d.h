#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

void unfolded3d_copy_kernel_cpu(
    Tensor& finput,
    Tensor& input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width);
void unfolded3d_acc_kernel_cpu(
    Tensor& finput,
    Tensor& input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width);

} // namespace native
} // namespace at
