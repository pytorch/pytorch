#pragma once
#include <detail/oneDNN.h>

namespace at {
namespace native::xpu {
xpu::onednn::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    xpu::onednn::Attr attr);

xpu::onednn::Attr string_to_unary_attr(xpu::onednn::Attr attr);

xpu::onednn::Attr construct_unary_attr(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    xpu::onednn::Attr attr);

xpu::onednn::Attr construct_binary_attr(
    c10::string_view binary,
    c10::optional<at::Scalar> alpha,
    const Tensor& other,
    xpu::onednn::Attr attr);

} // namespace native::xpu
} // namespace at
