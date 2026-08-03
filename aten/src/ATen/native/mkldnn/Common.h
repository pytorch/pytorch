#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>
#include <utility>

namespace at::native::mkldnn {

struct ContextConv final {
  ideep::tensor weight_packed_;
  std::optional<at::Tensor> at_bias_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  ideep::attr_t attr_;

  ContextConv() = delete;

  ContextConv(
      ideep::tensor&& weight_packed,
      std::optional<at::Tensor> at_bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups,
      ideep::attr_t attr)
      : weight_packed_(std::move(weight_packed)),
        at_bias_(std::move(at_bias)),
        padding_(std::move(padding)),
        stride_(std::move(stride)),
        dilation_(std::move(dilation)),
        groups_(groups),
        attr_(attr) {}
};

} // namespace at

#endif // AT_MKLDNN_ENABLED()
