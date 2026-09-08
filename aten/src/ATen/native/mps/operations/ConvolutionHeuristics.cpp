#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/operations/ConvolutionHeuristics.h>
#include <c10/metal/common.h>

#include <algorithm>
#include <limits>

namespace at::native::mps {

// Estimate padded work adjusted for staging efficiency and device occupancy.
// Lower-cost tiles do less wasted work while keeping enough threadgroups active.
float Conv3dMppTile::cost(const Conv3DParams& params, int64_t groups, int64_t cores) const {
  constexpr float kOccupancyFill = 300.0f;
  constexpr float kSimdgroupOverhead = 8.0f;
  constexpr float kOutputChannelOverhead = 90.0f;
  constexpr float kWeightLoadCost = 0.15f;
  constexpr float kStagingOverhead = 0.4f;

  const auto& conv2d = params.conv2d;
  const auto output_channel_tiles =
      c10::metal::ceil_div(int64_t(conv2d.C_out_per_group), int64_t(output_channels));
  const auto width_tiles = c10::metal::ceil_div(int64_t(conv2d.outW), int64_t(output_width));
  const auto height_tiles = c10::metal::ceil_div(int64_t(conv2d.outH), int64_t(output_height));
  const auto threadgroups =
      int64_t(output_channel_tiles) * groups * width_tiles * height_tiles * conv2d.N * params.outD;
  const auto kernel_elements = params.kD * conv2d.kH * conv2d.kW;
  const auto padded_output = float(threadgroups) * output_channels * output_width * output_height;

  const auto outputs_per_simdgroup = float(output_width * output_height) / simdgroups;
  const auto input_tile_width =
      int64_t(output_width - 1) * conv2d.sW + (conv2d.kW - 1) * conv2d.dW + 1;
  const auto input_tile_height =
      int64_t(output_height - 1) * conv2d.sH + (conv2d.kH - 1) * conv2d.dH + 1;
  const auto staging_ratio = float(output_width * output_height) /
      (float(input_tile_width * input_tile_height) + kWeightLoadCost * output_channels * kernel_elements);
  const auto tile_efficiency = outputs_per_simdgroup / (outputs_per_simdgroup + kSimdgroupOverhead) *
      (output_channels / (output_channels + kOutputChannelOverhead)) *
      (staging_ratio / (staging_ratio + kStagingOverhead));
  const auto device_utilization = float(threadgroups) / (threadgroups + cores * kOccupancyFill);
  return padded_output / (tile_efficiency * device_utilization);
}

Conv3dMppTile select_conv3d_mpp_tile(const Conv3DParams& params, int64_t groups) {
  static const int64_t cores = []() {
    const unsigned device_cores = at::mps::MPSDevice::getInstance()->getCoreCount();
    return device_cores > 0 ? static_cast<int64_t>(device_cores) : 16;
  }();
  // Keep in sync with INSTANTIATE_CONV3D_MPP_TILES in Convolution.metal.
  const Conv3dMppTile candidates[] = {
      {.output_channels = 32, .output_width = 8, .output_height = 8, .simdgroups = 2},
      {.output_channels = 32, .output_width = 16, .output_height = 8, .simdgroups = 4},
      {.output_channels = 64, .output_width = 8, .output_height = 8, .simdgroups = 2},
      {.output_channels = 64, .output_width = 16, .output_height = 8, .simdgroups = 4},
      {.output_channels = 128, .output_width = 8, .output_height = 8, .simdgroups = 4},
  };
  auto best_tile = candidates[0];
  auto best_cost = std::numeric_limits<float>::max();
  for (const auto& candidate : candidates) {
    const auto candidate_cost = candidate.cost(params, groups, cores);
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_tile = candidate;
    }
  }
  return best_tile;
}

Conv3dSimdTile select_conv3d_simd_tile(int64_t output_height, int64_t output_width, bool use_long_index) {
  if (use_long_index) {
    return {.block_rows = 64, .block_columns = 64, .simdgroups_rows = 2, .simdgroups_columns = 2};
  }
  if (output_height * output_width < 48) {
    return {.block_rows = 32, .block_columns = 64, .simdgroups_rows = 1, .simdgroups_columns = 2};
  }
  return {.block_rows = 64, .block_columns = 64, .simdgroups_rows = 2, .simdgroups_columns = 2};
}

// Materializing im2col only wins for patch embeddings and occupancy-starved
// single-batch convolutions with a large reduction.
bool conv3d_prefer_im2col(const Tensor& input,
                          const Tensor& weight,
                          IntArrayRef stride,
                          IntArrayRef padding,
                          IntArrayRef dilation,
                          int64_t groups,
                          const Tensor& output) {
  if (groups != 1 || dilation[0] != 1 || dilation[1] != 1 || dilation[2] != 1) {
    return false;
  }
  const int64_t kernel_depth = weight.size(2);
  const int64_t kernel_height = weight.size(3);
  const int64_t kernel_width = weight.size(4);
  const int64_t reduction_size = weight.size(1) * kernel_depth * kernel_height * kernel_width;

  // The long-indexed direct kernel owns oversized planes, and the im2col
  // matrix must fit MPSGraph's int32 element limit.
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  const int64_t output_positions = output.size(0) * output.size(2) * output.size(3) * output.size(4);
  if (input.size(1) * input.size(3) * input.size(4) > kInt32Max ||
      output.size(1) * output.size(3) * output.size(4) > kInt32Max ||
      output_positions > kInt32Max / std::max<int64_t>(reduction_size, 1)) {
    return false;
  }
  const bool patch_embedding = padding[0] == 0 && padding[1] == 0 && padding[2] == 0 &&
      stride[0] == kernel_depth && stride[1] == kernel_height && stride[2] == kernel_width;
  if (patch_embedding) {
    return reduction_size >= 256;
  }
  const int64_t output_volume = output.size(2) * output.size(3) * output.size(4);
  return input.scalar_type() == kFloat && input.size(0) == 1 && output_volume <= 256 && reduction_size >= 4096;
}

} // namespace at::native::mps
