#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace at::native {

TORCH_API Tensor NestedTensor_to_padded_tensor_generic(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size);

template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_sizes();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}
template <typename Func>
Tensor map_nt_binary(const Tensor& nt_1, const Tensor& nt_2, Func f){
  auto* nt_impl_1 = get_nested_tensor_impl(nt_1);
  auto* nt_impl_2 = get_nested_tensor_impl(nt_2);
  const auto& sizes = nt_impl_1->get_nested_sizes();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl_1->get_buffer(), nt_impl_2->get_buffer()), sizes);
}

C10_ALWAYS_INLINE std::pair<int64_t, int64_t> _check_nested_layer_norm_inputs(
    const NestedTensorImpl& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */) {

  const size_t normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  // Check that the normalized_shape has the exact same sizes as the last dimensions from the NestedTensor input
  // Also, compute M and N considering the idiosyncracies of NestedTensors
  int64_t N = 1;
  for (const auto i: c10::irange(normalized_ndim)) {
    TORCH_CHECK(
      input.opt_size(-normalized_ndim + i).has_value(),
      "normalized_shape extends into irregular dimensions for the nested tensor"
    );
    TORCH_CHECK(
      normalized_shape[i] == input.opt_size(-normalized_ndim + i),
      "The shape at dimension ",
      i,
      "of normalized_shape doesn't match the input"
    );
    N *= normalized_shape[i];
  }

  const int64_t M = input.numel() / N;

  return std::make_pair(M, N);
}

Tensor reshape_nested(const Tensor& self, IntArrayRef proposed_shape);

} // namespace at::native
