#pragma once

#include <ATen/core/ivalue.h>

struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) = 0;

  virtual at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) = 0;
  virtual at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) = 0;

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual c10::optional<at::Tensor> bias() = 0;

  virtual void set_bias(c10::optional<at::Tensor> bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }
};
