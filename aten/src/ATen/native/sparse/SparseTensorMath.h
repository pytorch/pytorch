#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

TORCH_API const sparse::SparseTensor& mul_out_sparse_scalar(const sparse::SparseTensor& r, const sparse::SparseTensor& t, const Scalar& value);
TORCH_API const sparse::SparseTensor& mul_out_sparse_zerodim(const sparse::SparseTensor& r, const sparse::SparseTensor& t, const Tensor& value);

}}
