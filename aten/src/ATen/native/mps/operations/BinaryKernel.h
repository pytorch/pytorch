#pragma once

namespace at::native::mps {
void binary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
void complex_mul_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
} // namespace at::native::mps
