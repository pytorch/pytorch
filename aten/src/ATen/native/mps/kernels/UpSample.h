#pragma once
#include <c10/metal/common.h>

#ifndef __METAL_VERSION__
#include <ATen/native/UpSample.h>
#include <initializer_list>
#include <optional>
#endif

template <unsigned N = 5>
struct UpsampleParams {
  ::c10::metal::array<int64_t, N> input_strides;
  ::c10::metal::array<int64_t, N> input_sizes;
  ::c10::metal::array<int64_t, N> output_strides;
  ::c10::metal::array<int64_t, N> output_sizes;
  ::c10::metal::array<float, N - 2> scales;
  bool align_corners;

#ifndef __METAL_VERSION__
  // Host-side fill. Geometry comes straight from the tensors (no narrowing --
  // the struct stores int64, matching Tensor::sizes()/strides()). The kernel
  // "scale" is the input/output ratio; `spatial_scales` are the optional
  // user-supplied scale factors, innermost-first (w, h, d), consumed up to the
  // N-2 spatial dims. For the backward pass, pass grad_input as `input` and
  // grad_output as `output`.
  UpsampleParams(
      const at::TensorBase& input,
      const at::TensorBase& output,
      bool align_corners_,
      std::initializer_list<std::optional<double>> spatial_scales)
      : align_corners(align_corners_) {
    for (unsigned dim = 0; dim < N; dim++) {
      input_sizes[dim] = input.size(dim);
      input_strides[dim] = input.stride(dim);
      output_sizes[dim] = output.size(dim);
      output_strides[dim] = output.stride(dim);
    }
    auto scale_it = spatial_scales.begin();
    for (unsigned k = 0; k < N - 2; k++, ++scale_it) {
      const auto dim = N - 1 - k;
      scales[k] = at::native::area_pixel_compute_scale<float>(
          input.size(dim), output.size(dim), align_corners_, *scale_it);
    }
  }
#endif
};
