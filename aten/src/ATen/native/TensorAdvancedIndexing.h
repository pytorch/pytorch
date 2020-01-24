#pragma once

// Indexing tensors by by tensors

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

using index_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
using index_put_accum_fn = void(*)(Tensor &, TensorList , const Tensor &, bool unsafe);

using gather_fn = void (*)(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
using scatter_add_fn = void(*)(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(index_put_accum_fn, index_put_accum_stub);

DECLARE_DISPATCH(gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);

}} // namespace at::native
