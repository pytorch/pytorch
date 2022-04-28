#pragma once

#include <ATen/ATen.h>
#include <ideep/tensor.hpp>

namespace at {
namespace native {
namespace mkldnn {

struct ContextConv2D final {
  ideep::tensor weight_packed_;
  c10::optional<at::Tensor> at_bias_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  int64_t groups_;
  ideep::attr_t attr_;

  ContextConv2D() = delete;

  ContextConv2D(
      ideep::tensor&& weight_packed,
      c10::optional<at::Tensor> at_bias,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups,
      ideep::attr_t attr)
      : weight_packed_(std::move(weight_packed)),
        at_bias_(std::move(at_bias)),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        attr_(attr) {}
};

} // namespace mkldnn
} // namespace native
} // namespace at
