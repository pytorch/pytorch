#pragma once

#include <ATen/core/Tensor.h>

namespace at::native {

void mxfp_mm_emulated(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const Tensor& bias_data,
    Tensor& out);

} // namespace at::native
