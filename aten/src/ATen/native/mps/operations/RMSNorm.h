#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

namespace at::native::mps {

Tensor rms_norm_mps_kernel(
    const Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const Tensor& weight,
    const double eps);

} // namespace at::native::mps
