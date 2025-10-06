#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>

namespace at::native::xpu {
class QConvoneDNNXPU final {
 public:
  C10_API static at::Tensor run_pointwise(
      at::Tensor act,
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double inv_output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm);

  C10_API static at::Tensor run_pointwise_tensor(
      at::Tensor act,
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<std::string_view> algorithm);

  C10_API static at::Tensor run_pointwise_binary(
      at::Tensor act,
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm);

  C10_API static at::Tensor run_pointwise_binary_tensor(
      at::Tensor act,
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      at::Tensor accum,
      std::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double accum_scale,
      int64_t accum_zero_point,
      std::string_view binary_attr,
      std::optional<at::Scalar> alpha,
      std::optional<std::string_view> unary_attr,
      torch::List<std::optional<at::Scalar>> unary_scalars,
      std::optional<std::string_view> unary_algorithm);

  static inline c10::ScalarType qconv_decide_out_dtype(
      const at::Tensor& act,
      const std::optional<c10::ScalarType> output_dtype);

  static at::Tensor qconv_prepack_xpu(
      at::Tensor weight,
      at::Tensor weight_scales,
      double input_scale,
      int64_t input_zero_point,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      std::optional<torch::List<int64_t>> input_shape);
};

} // namespace at::native::xpu