#pragma once

namespace at::native::mps {
void binary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other,
    const Tensor& output,
    const std::optional<Scalar> alpha = std::nullopt);
} // namespace at::native::mps
