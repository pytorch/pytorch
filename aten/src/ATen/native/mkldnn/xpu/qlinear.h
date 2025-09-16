#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>

namespace at::native::xpu {

class QLinearOnednnXPU final {
 public:
  C10_API static Tensor q_linear_pointwise(
      Tensor act,
      double act_scale,
      int64_t act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      std::string_view post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_tensor(
      Tensor act,
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      std::string_view post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_binary(
      Tensor act,
      double act_scale,
      int64_t act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      std::string_view binary_post_op,
      double binary_alpha,
      std::string_view unary_post_op,
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      std::string_view unary_post_op_algorithm);

  C10_API static Tensor q_linear_pointwise_binary_tensor(
      Tensor act,
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor weight,
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      std::string_view binary_post_op,
      double binary_alpha,
      std::string_view unary_post_op,
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      std::string_view unary_post_op_algorithm);

  C10_API static Tensor q_linear_prepack_onednn(
      at::Tensor weight,
      std::optional<torch::List<int64_t>> input_shape);

  static inline c10::ScalarType qlinear_decide_out_dtype(
      const at::Tensor& act,
      const std::optional<c10::ScalarType> output_dtype);

}; // class QLinearOnednnXPU

} // namespace at::native::xpu
