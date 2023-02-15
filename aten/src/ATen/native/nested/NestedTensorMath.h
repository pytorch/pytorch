#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}
template <typename Func>
Tensor map_nt_binary(const Tensor& nt_1, const Tensor& nt_2, Func f){
  auto* nt_impl_1 = get_nested_tensor_impl(nt_1);
  auto* nt_impl_2 = get_nested_tensor_impl(nt_2);
  const auto& sizes = nt_impl_1->get_nested_size_tensor();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl_1->get_buffer(), nt_impl_2->get_buffer()), sizes);
}

} // namespace native
} // namespace at
