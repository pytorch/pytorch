#pragma once

// Indexing tensors by by tensors

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

enum class SCATTER_GATHER_OP: uint8_t {REDUCE_ADD, REDUCE_MULTIPLY};

using index_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_fill_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride, const Scalar& source);
using index_copy_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
using index_put_with_sort_fn = void(*)(Tensor &, const c10::List<c10::optional<Tensor>> &, const Tensor &, bool accumulate, bool unsafe);
using masked_fill_fn = void(*)(TensorIterator &, const Scalar& scalar);
using put_fn = void(*)(TensorIterator & iter, const Tensor& self, const bool accumulate);
using take_fn = void(*)(TensorIterator & iter, const Tensor& input);
using masked_select_fn = void(*)(TensorIterator &, int64_t orig_stride);
using masked_scatter_fn = void(*)(TensorIterator &, const Tensor &);

using gather_fn = void (*)(const Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
using scatter_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_fill_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& src);
using scatter_add_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                  const Tensor& src, const SCATTER_GATHER_OP& reduce);
using scatter_scalar_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                         const Scalar& value, const SCATTER_GATHER_OP& reduce);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_fill_fn, index_fill_stub);
DECLARE_DISPATCH(index_copy_fn, index_copy_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub);
DECLARE_DISPATCH(put_fn, put_stub);
DECLARE_DISPATCH(take_fn, take_stub);
DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_serial_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_stub);
DECLARE_DISPATCH(masked_scatter_fn, masked_scatter_stub);

DECLARE_DISPATCH(gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_fn, scatter_stub);
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub);
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub);

TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, const c10::List<c10::optional<at::Tensor>>& indices);

}} // namespace at::native
