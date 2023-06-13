#pragma once

// Indexing tensors by tensors

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/cpu/radix_sort.h>

namespace at {
struct TensorIterator;
}

namespace at { namespace native {

using index_put_with_sort_fn = void(*)(Tensor &, const c10::List<c10::optional<Tensor>> &, const Tensor &, bool accumulate, bool unsafe);
using index_put_with_sort_quantized_fn = void(*)(Tensor& self, const c10::List<c10::optional<Tensor>>& indices, const Tensor& value, double scale, int zero_point, bool unsafe);
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

DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub);
DECLARE_DISPATCH(index_put_with_sort_quantized_fn, index_put_with_sort_quantized_stub);
DECLARE_DISPATCH(gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_fn, scatter_stub);
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub);
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub);
DECLARE_DISPATCH(scatter_reduce_two_fn, scatter_reduce_two_stub);

TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, const c10::List<c10::optional<at::Tensor>>& indices);

// fast paths for GNN usage
static inline bool can_use_expanded_index_path(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    bool is_scatter_like) {
  if (!self.device().is_cpu()) {
    return false;
  }

  const auto st = self.scalar_type();
  if (!(c10::isFloatingType(st)) || st == ScalarType::Half) {
    return false;
  }

  if (!is_radix_sort_available()) {
    return false;
  }

  // skip when having empty tensor
  if (self.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return false;
  }

  // skip when having scalar tensor
  if (self.ndimension() == 0 || index.ndimension() == 0 || src.ndimension() == 0) {
    return false;
  }

  // allow only different size on dim 0 for src and index
  // https://github.com/pytorch/pytorch/issues/99595
  for (const auto dim : c10::irange(1, index.dim())) {
    if (src.size(dim) != index.size(dim)) {
      return false;
    }
  }

  if (is_scatter_like) {
    // using `spmm` for scatter would require sorting on index,
    // this is only perf beneficial when the inner dimension, aka, `channels`
    // is big enough.
    constexpr int64_t threshold = 16;
    if (index.numel() / index.size(0) < threshold) {
      return false;
    }
  }

  // usually the expanded index has stride on the first dimension to be 1,
  // and strides on other dims to be 0 or 1, e.g.
  //   shape [108365, 16]; strides [1, 0]
  //   shape [13264, 1, 7]; strides [1, 1, 0]
  auto index_strides = index.strides().vec();
  bool is_index_expanded = index_strides[0] == 1;
  for (const auto dim : c10::irange(1, index_strides.size())) {
    if (index_strides[dim] > 1) { is_index_expanded = false; }
  }

  // index is expanded
  return dim == 0 && is_index_expanded && src.is_contiguous() && self.is_contiguous();
}

using scatter_add_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&);
using scatter_reduce_expanded_index_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const ReductionType& reduce, bool);
using gather_expanded_index_fn = void (*)(const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(scatter_add_expanded_index_fn, scatter_add_expanded_index_stub);
DECLARE_DISPATCH(scatter_reduce_expanded_index_fn, scatter_reduce_expanded_index_stub);
DECLARE_DISPATCH(gather_expanded_index_fn, gather_expanded_index_stub);

}} // namespace at::native
