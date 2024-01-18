#include <ATen/ATen.h>

namespace at {
namespace native {

// alias for to_padded_tensor in nested namespace
Tensor nested_to_padded_tensor(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
    return t.to_padded_tensor(padding, output_size);
}

Tensor alias_nested(const Tensor& self) {
  auto* nt_impl = get_nested_tensor_impl(self);
  const at::Tensor& buffer = self_ptr->get_unsafe_storage_as_tensor();
  const auto& nested_sizes = nt_impl->get_nested_sizes();
  const auto& nested_strides = nt_impl->get_nested_strides();
  const auto& storage_offsets = self_ptr->get_storage_offsets();
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(buffer),
      std::move(nested_sizes),
      std::move(nested_strides),
      std::move(storage_offsets));
}

} // namespace native
} // namespace at
