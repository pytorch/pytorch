#pragma once

// Indexing tensors by tensors

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>

namespace at {
struct TensorIterator;
}

namespace at::native {

using index_put_with_sort_fn = void(*)(Tensor &, const c10::List<std::optional<Tensor>> &, const Tensor &, bool accumulate, bool unsafe);
using index_put_with_sort_quantized_fn = void(*)(Tensor& self, const c10::List<std::optional<Tensor>>& indices, const Tensor& value, double scale, int zero_point, bool unsafe);
using gather_fn = void (*)(const Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
using scatter_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_fill_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& src);
using scatter_add_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                  const Tensor& src, const ReductionType& reduce);
using scatter_scalar_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                         const Scalar& value, const ReductionType& reduce);
using scatter_reduce_two_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                      const Tensor& src, const ReductionType& reduce);

DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub)
DECLARE_DISPATCH(index_put_with_sort_quantized_fn, index_put_with_sort_quantized_stub)
DECLARE_DISPATCH(gather_fn, gather_stub)
DECLARE_DISPATCH(scatter_fn, scatter_stub)
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub)
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub)
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub)
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub)
DECLARE_DISPATCH(scatter_reduce_two_fn, scatter_reduce_two_stub)

TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, const c10::List<std::optional<at::Tensor>>& indices);

using scatter_add_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&);
using scatter_reduce_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const ReductionType& reduce, bool);
using gather_expanded_index_fn = void (*)(const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(scatter_add_expanded_index_fn, scatter_add_expanded_index_stub)
DECLARE_DISPATCH(scatter_reduce_expanded_index_fn, scatter_reduce_expanded_index_stub)
DECLARE_DISPATCH(gather_expanded_index_fn, gather_expanded_index_stub)

} // namespace at::native
