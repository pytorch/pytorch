#pragma once
#include <ATen/core/Tensor.h>

namespace at::native::mps {

void linalg_solve_triangular_mps_impl(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool transpose,
    bool left,
    bool unitriangular,
    const Tensor& out,
    bool conjugate = false);

} // namespace at::native::mps
