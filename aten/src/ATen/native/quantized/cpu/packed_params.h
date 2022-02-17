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

  // out variant of LinearPackedParamsBase::apply
  virtual at::Tensor& apply_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual at::Tensor& apply_relu_out(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point,
      at::Tensor& output) {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual at::Tensor apply_dynamic(
      at::Tensor input,
      bool reduce_range = false) = 0;
  virtual at::Tensor apply_dynamic_relu(
      at::Tensor input,
      bool reduce_range = false) = 0;

  virtual at::Tensor& apply_dynamic_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_out is not implemented for this packed "
        "parameter type");
    return output;
  }
  virtual at::Tensor& apply_dynamic_relu_out(
      const at::Tensor& /* input */,
      at::Tensor& output,
      bool /* reduce_range */) {
    throw std::runtime_error(
        "apply_dynamic_relu_out is not implemented for this packed "
        "parameter type");
    return output;
  }

  virtual std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() = 0;

  virtual c10::optional<at::Tensor> bias() = 0;

  virtual void set_bias(c10::optional<at::Tensor> bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }
};
