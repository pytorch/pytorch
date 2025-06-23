#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::xpu {
C10_API Tensor q_linear_pointwise_tensor(
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

} // namespace at::native::xpu

#endif // AT_MKLDNN_ENABLED()
