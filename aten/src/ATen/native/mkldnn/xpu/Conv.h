#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::xpu {
C10_API Tensor convolution_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm);

C10_API Tensor convolution_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm);

C10_API Tensor& convolution_pointwise_binary_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string_view> unary_algorithm);

} // namespace at::native::xpu

#endif // AT_MKLDNN_ENABLED()
