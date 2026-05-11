#pragma once
#include <c10/metal/common.h>

#ifdef __METAL_VERSION__
// Must match at::native::detail::GridSamplerInterpolation
enum class GridSamplerInterpolation : int32_t {
  Bilinear = 0,
  Nearest = 1,
  Bicubic = 2,
};

// Must match at::native::detail::GridSamplerPadding
enum class GridSamplerPadding : int32_t {
  Zeros = 0,
  Border = 1,
  Reflection = 2,
};
#else
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/Pool.h>
using at::native::GridSamplerInterpolation;
using at::native::GridSamplerPadding;
#endif

template <unsigned N = 5, typename idx_t = int32_t>
struct GridSamplerParams {
  int32_t sampler_dims;
  ::c10::metal::array<idx_t, N> output_sizes;
  ::c10::metal::array<idx_t, N> output_strides;
  ::c10::metal::array<idx_t, N> input_sizes;
  ::c10::metal::array<idx_t, N> input_strides;
  ::c10::metal::array<idx_t, N> grid_sizes;
  ::c10::metal::array<idx_t, N> grid_strides;
  bool align_corners;

#ifndef __METAL_VERSION__
  GridSamplerParams(
      const at::TensorBase& output,
      const at::TensorBase& input,
      const at::TensorBase& grid,
      bool align_corners_)
      : sampler_dims(N - 2), align_corners(align_corners_) {
    for (unsigned dim = 0; dim < N; dim++) {
      output_sizes[dim] =
          c10::checked_convert<idx_t>(output.size(dim), "int32_t");
      output_strides[dim] =
          c10::checked_convert<idx_t>(output.stride(dim), "int32_t");
      input_sizes[dim] =
          c10::checked_convert<idx_t>(input.size(dim), "int32_t");
      input_strides[dim] =
          c10::checked_convert<idx_t>(input.stride(dim), "int32_t");
      grid_sizes[dim] = c10::checked_convert<idx_t>(grid.size(dim), "int32_t");
      grid_strides[dim] =
          c10::checked_convert<idx_t>(grid.stride(dim), "int32_t");
    }
  }
#endif
};

template <unsigned N = 5, typename idx_t = int32_t>
struct GridSamplerBackwardParams {
  GridSamplerParams<N, idx_t> forward;
  ::c10::metal::array<idx_t, N> grad_output_strides;
  ::c10::metal::array<idx_t, N> grad_input_strides;
  ::c10::metal::array<idx_t, N> grad_grid_strides;
  GridSamplerPadding padding_mode;
  GridSamplerInterpolation interpolation_mode;
  bool compute_grad_input;
  bool compute_grad_grid;

#ifndef __METAL_VERSION__
  GridSamplerBackwardParams(
      const at::TensorBase& grad_output,
      const at::TensorBase& input,
      const at::TensorBase& grid,
      const at::TensorBase& grad_input,
      const at::TensorBase& grad_grid,
      bool align_corners,
      GridSamplerPadding padding_mode_,
      GridSamplerInterpolation interpolation_mode_)
      : forward(grad_output, input, grid, align_corners),
        padding_mode(padding_mode_),
        interpolation_mode(interpolation_mode_),
        compute_grad_input(grad_input.defined()),
        compute_grad_grid(
            interpolation_mode_ != GridSamplerInterpolation::Nearest) {
    for (unsigned dim = 0; dim < N; dim++) {
      grad_output_strides[dim] =
          c10::checked_convert<idx_t>(grad_output.stride(dim), "int32_t");
      grad_input_strides[dim] = grad_input.defined()
          ? c10::checked_convert<idx_t>(grad_input.stride(dim), "int32_t")
          : 0;
      grad_grid_strides[dim] =
          c10::checked_convert<idx_t>(grad_grid.stride(dim), "int32_t");
    }
  }
#endif
};
