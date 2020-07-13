#pragma once

#include <array>

#include <ATen/Tensor.h>

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

namespace at {
namespace native {
namespace vulkan {

struct ContextConv2D final {
  at::Tensor weight_prepacked_vulkan_;
  c10::optional<at::Tensor> bias_vulkan_;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  int64_t groups_;
  float output_min_;
  float output_max_;

  ContextConv2D() = delete;

  ContextConv2D(
      at::Tensor&& weight_prepacked_vulkan,
      c10::optional<at::Tensor>&& bias_vulkan,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups,
      float output_min,
      float output_max)
      : weight_prepacked_vulkan_(std::move(weight_prepacked_vulkan)),
        bias_vulkan_(std::move(bias_vulkan)),
        weight_size_(weight_size),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        output_min_(output_min),
        output_max_(output_max) {}

  ContextConv2D(ContextConv2D&&) = default;
  ContextConv2D& operator=(ContextConv2D&&) = default;

  ~ContextConv2D() {}

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

} // namespace vulkan
} // namespace native
} // namespace at
