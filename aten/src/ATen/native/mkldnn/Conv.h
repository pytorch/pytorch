#pragma once

#include <ATen/Tensor.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
C10_API Tensor mkldnn_convolution_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::string attr,
    std::vector<std::optional<at::Scalar>> scalars,
    std::optional<std::string> algorithm);

C10_API Tensor mkldnn_convolution_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::string binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string> unary_attr,
    std::vector<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string> unary_algorithm);

C10_API Tensor& mkldnn_convolution_pointwise_binary_(
    Tensor& other_t,
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::string binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string> unary_attr,
    std::vector<std::optional<at::Scalar>> unary_scalars,
    std::optional<std::string> unary_algorithm);

Tensor mkldnn_convolution_transpose_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::string attr,
    std::vector<std::optional<at::Scalar>> scalars,
    std::optional<std::string> algorithm);

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
