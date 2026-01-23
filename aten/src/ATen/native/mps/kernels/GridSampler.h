#pragma once
#include <c10/metal/common.h>

#ifdef __METAL__
enum class GridSamplerInterpolation { Bilinear, Nearest, Bicubic };
enum class GridSamplerPadding { Zeros, Border, Reflection };
#else
#include <ATen/native/GridSamplerUtils.h>
using at::native::GridSamplerInterpolation;
using at::native::GridSamplerPadding;
#endif

template <unsigned N = 5, typename idx_type_t = int32_t>
struct GridSamplerParams {
  int32_t sampler_dims;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> grid_sizes;
  ::c10::metal::array<idx_type_t, N> grid_strides;
  GridSamplerInterpolation interpolation_mode;
  GridSamplerPadding padding_mode;
  bool align_corners;
};
