#pragma once

#include <ATen/Tensor.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
C10_API Tensor mkldnn_linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string attr,
    std::vector<std::optional<at::Scalar>> scalars,
    std::optional<std::string> algorithm);

C10_API Tensor mkldnn_linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string attr);

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
