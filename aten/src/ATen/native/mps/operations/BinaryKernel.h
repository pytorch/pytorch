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
void add_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
void sub_out(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output);
}
