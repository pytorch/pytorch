#pragma once

#include <ATen/Parallel.h>

namespace at {
namespace native {

template <typename Fn>
void dim_apply(TensorList tensors, int64_t dim, Fn f) {
  AT_ASSERT(tensors.size() > 0);
  auto t = tensors[0];
  auto sizes = t.sizes();
  int64_t ndim = t.dim();
  int64_t itersize = 1;
  for (int64_t i = 0; i < ndim; i++) {
    if (i != dim) {
      itersize *= t.size(i);
    }
  }
  parallel_for(0, itersize, 1, [&](int64_t i_begin, int64_t i_end) {
    std::vector<Tensor> narrowed_tensors;
    narrowed_tensors.reserve(tensors.size());
    for (int64_t it = i_begin; it < i_end; it++) {
      narrowed_tensors.clear();
      for (auto ti : tensors) {
        int64_t i = it;
        Tensor nt = ti;
        for (int64_t d = 0; d < ndim; d++) {
          if (d != dim) {
            // this could be avoided for slower-changing dimensions if done
            // better
            nt = nt.select((d > dim ? 1 : 0), i % sizes[d]);
            i = i / sizes[d];
          }
        }
        narrowed_tensors.emplace_back(nt);
      }
      f(it, narrowed_tensors);
    }
  });
}

// ensure we get good values and indices for kthvalue, mode, median
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
    if (!keepdim && indices.dim() == self.dim() - 1) {
      // unsqueeze to preserve passed in noncontiguous tensor in resize
      indices.unsqueeze_(dim);
    }
    indices.resize_(result_sizes);
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
