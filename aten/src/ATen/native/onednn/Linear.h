#pragma once

#include <ATen/Tensor.h>
#include <ATen/Config.h>

#if AT_ONEDNN_ENABLED()

namespace at {
namespace native {
C10_API Tensor onednn_linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::string_view attr,
    c10::List<std::optional<at::Scalar>> scalars,
    std::optional<c10::string_view> algorithm);

C10_API Tensor onednn_linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::string_view attr);

} // namespace native
} // namespace at

#endif // AT_ONEDNN_ENABLED()
