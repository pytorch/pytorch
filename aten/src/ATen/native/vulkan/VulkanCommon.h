#pragma once
#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

namespace at {
namespace native {
namespace vulkan {

struct Conv2DParams final {
  int64_t N;
  int64_t C;
  int64_t H;
  int64_t W;

  int64_t OC;
  int64_t KH;
  int64_t KW;

  int64_t SY;
  int64_t SX;
  int64_t PY;
  int64_t PX;
  int64_t DY;
  int64_t DX;
  int64_t G;
  int64_t OC_4;
  int64_t C_4;
  int64_t OW;
  int64_t OH;

  Conv2DParams() = delete;
  Conv2DParams(
      c10::IntArrayRef isizes,
      int64_t OC,
      int64_t KH,
      int64_t KW,
      int64_t SY,
      int64_t SX,
      int64_t PY,
      int64_t PX,
      int64_t DY,
      int64_t DX,
      int64_t G);

  Conv2DParams(
      c10::IntArrayRef inputSizes,
      c10::IntArrayRef weightSizes,
      c10::IntArrayRef padding,
      c10::IntArrayRef stride,
      c10::IntArrayRef dilation,
      int64_t G);

  std::vector<int64_t> output_sizes() {
    return {N, OC, OH, OW};
  }
};

struct ContextConv2D final {
  at::Tensor weight_prepacked_vulkan_;
  c10::optional<at::Tensor> bias_vulkan_;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  int64_t groups_;

  ContextConv2D() = delete;

  ContextConv2D(
      at::Tensor&& weight_prepacked_vulkan,
      c10::optional<at::Tensor>&& bias_vulkan,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups)
      : weight_prepacked_vulkan_(std::move(weight_prepacked_vulkan)),
        bias_vulkan_(std::move(bias_vulkan)),
        weight_size_(weight_size),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups) {}

  ContextConv2D(ContextConv2D&&) = default;
  ContextConv2D& operator=(ContextConv2D&&) = default;

  ~ContextConv2D() {}

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

} // namespace vulkan
} // namespace native
} // namespace at
