#pragma once
#include <detail/oneDNN.h>

//
// This header file provides utility functions for constructing and managing
// oneDNN attributes used in fusion operations on XPU devices. These utilities
// include functions for creating unary and binary post-operations attributes,
// as well as mapping string representations of operations to oneDNN attributes.
//

namespace at::native::xpu {
at::native::onednn::Attr& unary_attr_with_arg(
    onednn::Attr& attr,
    std::string_view unary,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm);

at::native::onednn::Attr& string_to_unary_attr(
    onednn::Attr& attr,
    std::string_view unary);

at::native::onednn::Attr& construct_unary_attr(
    onednn::Attr& attr,
    std::string_view unary,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm);

template <bool is_matmul = false>
onednn::Attr& construct_binary_attr(
    onednn::Attr& attr,
    std::string_view binary,
    const Tensor& other) {
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
  } else {
    TORCH_CHECK(
        binary == "none",
        "Binary attr ",
        binary,
        "is not supported for conv/linear post binary fusion");
  }
  return attr;
}

} // namespace at::native::xpu
