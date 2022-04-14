#pragma once

#include <ATen/NumericUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace native {

// ensure we get good values and indices for kthvalue, mode
// this will always be with the reducing dim as 1-d
inline void _reduction_with_indices_allocate_or_resize_output(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = 1;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    if (!keepdim && values.dim() == self.dim() - 1) {
      // unsqueeze to preserve passed in noncontiguous tensor in resize
      values.unsqueeze_(dim);
    }
    resize_output(values, result_sizes);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    if (!keepdim && indices.dim() == self.dim() - 1) {
      // unsqueeze to preserve passed in noncontiguous tensor in resize
      indices.unsqueeze_(dim);
    }
    resize_output(indices, result_sizes);
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
  }
}

// ensure we get good values and indices for topk
inline void _allocate_or_resize_output_with_indices(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    int64_t k) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto result_sizes = self.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  }
  if (values.defined()) {
    TORCH_CHECK(
        self.options().type_equal(values.options()),
        "output values must be of same type as input");
    values.resize_(result_sizes);
  } else {
    values = at::empty(result_sizes, self.options());
  }
  if (indices.defined()) {
    TORCH_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    TORCH_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as input");
    indices.resize_(result_sizes);
  } else {
    indices = at::empty(result_sizes, self.options().dtype(kLong));
  }
}

} // namespace native
} // namespace at
