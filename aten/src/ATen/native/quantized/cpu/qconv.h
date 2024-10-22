#pragma once
#include <ATen/Tensor.h>
#include <ATen/Config.h>

namespace at {
namespace native {

class QConvoneDNN final {
 public:

  C10_API static at::Tensor run_pointwise(
      at::Tensor act, // contains quantized values but not QTensor
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
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
      c10::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<c10::string_view> algorithm);

  C10_API static at::Tensor run_pointwise_tensor(
      at::Tensor act, // contains quantized values but not QTensor
      at::Tensor act_scale,
      at::Tensor act_zero_point,
      at::Tensor weight, // contains quantized values but not QTensor
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
      c10::string_view attr,
      torch::List<std::optional<at::Scalar>> scalars,
      std::optional<c10::string_view> algorithm);

};

} // namespace native
} // namespace at