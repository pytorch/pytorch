#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <optional>

namespace at {
namespace hip {
namespace detail {
void group_gemm_ck(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    at::Tensor& out);

} // namespace detail
} // namespace hip
} // namespace at
