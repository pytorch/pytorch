#pragma once
#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>

namespace at {
class Tensor;
class TensorBase;
struct TensorIterator;
struct TensorIteratorBase;
}

namespace c10 {
class Scalar;
}

namespace at { namespace native {

using index_fn = void(*)(TensorIteratorBase &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_fill_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride, const Scalar& source);
using index_copy_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
using put_fn = void(*)(TensorIterator & iter, const TensorBase& self, const bool accumulate);
using take_fn = void(*)(TensorIterator & iter, const TensorBase& input);
using flip_fn = void(*)(TensorIterator &, const bool);
using masked_fill_fn = void(*)(TensorIterator &, const Scalar& scalar);
using masked_select_fn = void(*)(TensorIterator &, int64_t orig_stride);
using masked_scatter_fn = void(*)(TensorIterator &, const TensorBase &);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_fill_fn, index_fill_stub);
DECLARE_DISPATCH(index_copy_fn, index_copy_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(put_fn, put_stub);
DECLARE_DISPATCH(take_fn, take_stub);
DECLARE_DISPATCH(flip_fn, flip_stub);
DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_serial_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_stub);
DECLARE_DISPATCH(masked_scatter_fn, masked_scatter_stub);

}} // namespace at::native
