#pragma once

#include <ATen/NumericUtils.h>
#include <ATen/native/Resize.h>

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


#ifdef CPU_CAPABILITY
inline namespace CPU_CAPABILITY {
#else
inline namespace DEFAULT {
#endif

// Core topk loop, shared between CPU and QuantizedCPU
template <typename scalar_t, typename accscalar_t>
void topk_impl_loop(
    const int64_t mode_values_stride,
    const int64_t mode_indices_stride,
    const int64_t tmp_values_stride,
    const int64_t k,
    const int64_t dim_size,
    const bool largest,
    const bool sorted,
    char** data, const int64_t* strides, const int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k, &mode_values_stride);
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k, &mode_indices_stride);
    TensorAccessor<scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size, &tmp_values_stride);

    auto n = dim_size;
    auto use_partial_sort = k * 64 <= n;

    using elem_t = std::pair<accscalar_t, int64_t>;
    std::vector<elem_t> queue(n);
    for (int64_t j = 0; j < n; j++) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    // we want nan to be sorted as top for numpy compatibility
    if (use_partial_sort) {
      if (largest) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
      } else {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
      }
    } else {
      if (largest) {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
            });
        }
      } else {
        std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k -1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
            });
        }
      }
    }

    for (int64_t j = 0; j < k; j++) {
      mode_values[j] = queue[j].first;
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace native
} // namespace at
