#pragma once

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>

namespace at {
namespace native {
namespace sparse {
namespace cuda {
namespace impl {

void addmm_out_sparse_csr_dense_cuda_impl(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result);

}
} // namespace cuda
} // namespace sparse
} // namespace native
} // namespace at
