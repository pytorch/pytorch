#pragma once
#include <detail/oneDNN.h>

namespace at::native::xpu {
at::native::onednn::Attr unary_attr_with_arg(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    onednn::Attr attr);

at::native::onednn::Attr string_to_unary_attr(onednn::Attr attr);

at::native::onednn::Attr construct_unary_attr(
    c10::string_view unary,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm,
    onednn::Attr attr);

template <bool is_matmul = false>
onednn::Attr construct_binary_attr(
    c10::string_view binary,
    c10::optional<at::Scalar> alpha,
    const Tensor& other,
    onednn::Attr attr) {
  if (binary == "mul") {
    attr.append_post_binary<is_matmul>(attr.kind_with_binary_mul, other);
  } else if (binary == "sub") {
    attr.append_post_binary<is_matmul>(attr.kind_with_binary_sub, other);
  } else if (binary == "div") {
    attr.append_post_binary<is_matmul>(attr.kind_with_binary_div, other);
  } else if (binary == "add") {
    attr.append_post_binary<is_matmul>(attr.kind_with_binary_add, other);
  } else if (binary == "sum") {
    attr.append_post_sum(1.f, 1.f, 0);
  }
  return attr;
}

} // namespace at::native::xpu
