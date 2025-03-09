#pragma once

namespace at::native::mps {
bool binary_alpha_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other,
    const Scalar& alpha,
    const Tensor& output);
void complex_mul_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
}
