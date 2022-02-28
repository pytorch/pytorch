#pragma once

#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

TORCH_API sparse::SparseTensor& mul_out_sparse_scalar(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Scalar& value);
TORCH_API sparse::SparseTensor& mul_out_sparse_zerodim(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Tensor& value);

}}
