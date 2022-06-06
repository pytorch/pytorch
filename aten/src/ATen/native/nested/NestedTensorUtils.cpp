#include <ATen/native/nested/NestedTensorUtils.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <tuple>

namespace at {
namespace native {

Tensor NestedTensor_to_buffer(const Tensor& self) {
  TORCH_CHECK(self.is_nested(), "Can only create a buffer from Nested Tensor");
  auto* nt_self = get_nested_tensor_impl(self);
  get_consistent_last_dim_of_nested_tensor(*nt_self);
  return nt_self->get_buffer().clone();
}

Tensor NestedTensor_from_buffer(const Tensor& buffer, const Tensor& shape) {
  TORCH_CHECK(
      !buffer.is_nested(),
      "Can only a create Nested Tensor from a normal tensor buffer");
  TORCH_CHECK(buffer.dim() == 1, "The input buffer must be flat");
  Tensor nt_buffer = buffer.clone();
  Tensor nt_size = shape.clone();
  return at::detail::make_tensor<NestedTensorImpl>(
      std::move(nt_buffer), std::move(nt_size));
}

Tensor NestedTensor_linear_composite(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  auto* nt_input = get_nested_tensor_impl_or_null(input);
  TORCH_CHECK(input.dim() == 3 && weight.dim() == 2);
  const auto last_dim = get_consistent_last_dim_of_nested_tensor(*nt_input);
  TORCH_CHECK(
      last_dim == weight.size(1),
      "shape mismatch for NestedTensor linear. NestedTensor last_dim: ",
      last_dim,
      " vs. first dim of rhs: ",
      weight.size(1));

  const Tensor& input_buffer = at::NestedTensor_to_buffer(input);
  Tensor result_buffer =
      at::linear(input_buffer.reshape({-1, weight.size(1)}), weight, bias_opt);
  result_buffer = result_buffer.reshape({-1});
  int64_t weight_size_1 = weight.size(0);

  // Calculated nested size
  Tensor new_sizes = nt_input->get_nested_size_tensor().clone();
  // Now the last entry in every row of new_sizes should be weight_size_1.
  new_sizes.index_put_({at::indexing::Slice(), -1}, weight_size_1);

  return at::NestedTensor_from_buffer(result_buffer, new_sizes);
}

} // namespace native
} // namespace at
