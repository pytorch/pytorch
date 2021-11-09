#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace mkl {

void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
