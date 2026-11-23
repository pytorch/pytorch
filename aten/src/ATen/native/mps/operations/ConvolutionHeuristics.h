#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/kernels/Convolution.h>

#include <cstdint>
#include <string>

namespace at::native::mps {

struct Conv3dMppTile {
  int output_channels;
  int output_width;
  int output_height;
  int simdgroups;

  float cost(const Conv3DParams& params, int64_t groups, int64_t cores) const;
};

struct Conv3dSimdTile {
  int block_rows;
  int block_columns;
  int simdgroups_rows;
  int simdgroups_columns;

  MTLComputePipelineState_t pipeline_state(
      MetalShaderLibrary& library,
      const std::string& dtype,
      bool use_long_index) const;
};

Conv3dMppTile select_conv3d_mpp_tile(
    const Conv3DParams& params,
    int64_t groups);
Conv3dSimdTile select_conv3d_simd_tile(
    int64_t output_height,
    int64_t output_width,
    bool use_long_index);
bool conv3d_prefer_im2col(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    const Tensor& output);

} // namespace at::native::mps
