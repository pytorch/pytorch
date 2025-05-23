#pragma once

namespace at::native::mps {
void unary_op_kernel(
    const std::string func_name,
    const Tensor& input,
    const Tensor& output);
} // namespace at::native::mps
