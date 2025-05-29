#pragma once

#include <ATen/Config.h>
#include <ATen/Tensor.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::xpu {
C10_API Tensor linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view attr,
    c10::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm);

C10_API Tensor linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view binary_attr);

} // namespace at::native::xpu

#endif // AT_MKLDNN_ENABLED()
