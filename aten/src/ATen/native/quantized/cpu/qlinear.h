#pragma once
#include <ATen/Tensor.h>
#include <ATen/Config.h>

namespace at::native {

class QLinearOnednn final {
 public:
  C10_API static Tensor run_pointwise_tensor(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      std::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      std::string_view post_op_algorithm);

C10_API static Tensor run_pointwise_binary_tensor(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other, // extra input for binary post-op
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      std::string_view binary_post_op, // e.g. "none", "sum", "add"
      double binary_alpha,
      std::string_view unary_post_op, // e.g. "none", "relu"
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      std::string_view unary_post_op_algorithm);
};

} // namespace at::native
