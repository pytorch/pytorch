#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace mkl {

void addmm_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result);

void triangular_solve_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular);

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
