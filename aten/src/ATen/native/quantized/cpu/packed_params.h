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

  virtual at::Tensor apply_dynamic(at::Tensor input) = 0;
  virtual at::Tensor apply_dynamic_relu(at::Tensor input) = 0;

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual std::string backend() = 0;
  virtual std::string bit_width() = 0;
};
