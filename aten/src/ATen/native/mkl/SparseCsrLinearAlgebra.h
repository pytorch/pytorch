#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/SparseCsrTensorUtils.h>

namespace at {
namespace sparse_csr {
Tensor& _sparse_mm_mkl_(
    Tensor& self,
    const SparseCsrTensor& sparse_,
    const Tensor& dense,
    const Tensor& t,
    const Scalar& alpha,
    const Scalar& beta);
} // namespace native
} // namespace at
