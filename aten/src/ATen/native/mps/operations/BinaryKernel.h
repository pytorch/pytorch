#pragma once

namespace at::native::mps {
void complex_mul_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
void real_mul_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
}
