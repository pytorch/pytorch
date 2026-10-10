#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <optional>

namespace at {
namespace hip {
namespace detail {

/**
 * @brief Tile the group gemm operation.
 * 
 * @param mat_a 
 * @param mat_b 
 * @param offs 
 * @param bias IGNORE THIS. SHOULD BE std::nullopt otherwise. Parent function that calls this crashes if not NULLOPT.  
 * @param out 
 */
void group_gemm_ck_tile(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    at::Tensor& out);
} // namespace detail
} // namespace hip
} // namespace at