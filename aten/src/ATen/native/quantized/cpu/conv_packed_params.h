#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

template <int kSpatialDim = 2>
struct ConvPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual torch::List<int64_t> stride() const = 0;
  virtual torch::List<int64_t> padding() const = 0;
  virtual torch::List<int64_t> dilation() const = 0;
  virtual int64_t groups() const = 0;
};
