#pragma once
#include <ATen/core/List.h>
#include <ATen/native/ConvUtils.h>

namespace at::native::quantized {
namespace {
// MakeConvOutputShape used from both CPU and CUDA libraries
// and exporting symbol from torch_cpu would probably take more storage
// than duplicating implementation which likely be inlined away
template <int kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, kSpatialDim>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

#if defined(USE_CUDA) || defined(USE_PYTORCH_QNNPACK)
template <>
at::SmallVector<int64_t, 4> MakeConvOutputShape<2>(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, 2>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const at::List<int64_t>& stride,
    const at::List<int64_t>& padding,
    const at::List<int64_t>& dilation) {
  const int H = input_image_shape[0];
  const int W = input_image_shape[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, M, Y_H, Y_W};
}

template <>
at::SmallVector<int64_t, 5> MakeConvOutputShape<3>(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, 3>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const at::List<int64_t>& stride,
    const at::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  const int D = input_image_shape[0];
  const int H = input_image_shape[1];
  const int W = input_image_shape[2];
  const int64_t Y_D =
      (D + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_H =
      (H + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  const int64_t Y_W =
      (W + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1;
  return {N, M, Y_D, Y_H, Y_W};
}

#endif
} // anonymous namespace
} // namespace at::native::quantized
