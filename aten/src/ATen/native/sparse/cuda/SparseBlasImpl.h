#pragma once

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace cuda {

void addmm_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

void addmv_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

void add_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const at::sparse_csr::SparseCsrTensor& mat2,
    const Scalar& alpha,
    const Scalar& beta,
    const at::sparse_csr::SparseCsrTensor& result);

void triangular_solve_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular);

} // namespace cuda
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
