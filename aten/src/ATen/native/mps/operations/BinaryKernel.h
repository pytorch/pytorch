#pragma once

namespace at::native::mps {
void binary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other,
    const Tensor& output,
    const std::optional<Scalar> alpha = std::nullopt,
    std::optional<uint32_t> ilp_threshold = std::nullopt);
void ternary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& other1,
    const Tensor& other2,
    const Tensor& output,
    std::optional<uint32_t> ilp_threshold = std::nullopt);
} // namespace at::native::mps
