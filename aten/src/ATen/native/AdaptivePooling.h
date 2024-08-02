#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <cmath>

namespace at::native {

using adaptive_avg_pooling2d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling2d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling2d_fn, adaptive_avg_pool2d_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling2d_backward_fn, adaptive_avg_pool2d_backward_kernel);

using adaptive_max_pooling2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
using adaptive_max_pooling2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
DECLARE_DISPATCH(adaptive_max_pooling2d_fn, adaptive_max_pool2d_kernel);
DECLARE_DISPATCH(adaptive_max_pooling2d_backward_fn, adaptive_max_pool2d_backward_kernel);

using adaptive_avg_pooling3d_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling3d_fn, adaptive_avg_pool3d_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling3d_backward_fn, adaptive_avg_pool3d_backward_kernel);

using adaptive_max_pooling3d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
using adaptive_max_pooling3d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
DECLARE_DISPATCH(adaptive_max_pooling3d_fn, adaptive_max_pool3d_kernel);
DECLARE_DISPATCH(adaptive_max_pooling3d_backward_fn, adaptive_max_pool3d_backward_kernel);

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

inline void adaptive_pool_empty_output_check(const Tensor& gradOutput_, const char* arg_name) {
  int64_t ndim = gradOutput_.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(gradOutput_.size(i) > 0,
      arg_name, "(): Expected grad_output to have non-zero size for non-batch dimensions, "
      "but grad_output has sizes ", gradOutput_.sizes(), " with dimension ", i,
      " being empty");
  }
}

} // namespace at::native
