#include "native/Extra.h"

#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>

#include <torch/library.h>

namespace at::openreg {

at::Tensor wrapper_quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor_openreg(
      self, scale, zero_point, dtype);
}

int64_t wrapper__fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  return at::native::_fused_sdp_choice_openreg(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
wrapper__scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  return at::native::_scaled_dot_product_fused_attention_overrideable_openreg(
      query,
      key,
      value,
      attn_bias,
      dropout_p,
      is_causal,
      return_debug_mask,
      scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
wrapper_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return at::native::
      _scaled_dot_product_fused_attention_overrideable_backward_openreg(
          grad_out,
          query,
          key,
          value,
          attn_bias,
          grad_input_mask,
          out,
          logsumexp,
          cum_seq_q,
          cum_seq_k,
          max_q,
          max_k,
          dropout_p,
          is_causal,
          philox_seed,
          philox_offset,
          scale);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("quantize_per_tensor", &wrapper_quantize_per_tensor);
  m.impl("_fused_sdp_choice", &wrapper__fused_sdp_choice);
  m.impl(
      "_scaled_dot_product_fused_attention_overrideable",
      &wrapper__scaled_dot_product_fused_attention_overrideable);
  m.impl(
      "_scaled_dot_product_fused_attention_overrideable_backward",
      &wrapper_scaled_dot_product_fused_attention_overrideable_backward);
}

} // namespace at::openreg

namespace at::openreg {
TORCH_LIBRARY(openreg, m) {
  m.def("custom_autograd_fn_returns_self(Tensor input)-> Tensor");
  m.def("custom_autograd_fn_aliasing(Tensor(a) input)-> Tensor(a)");
}

TORCH_LIBRARY_IMPL(openreg, AutogradPrivateUse1, m) {
  m.impl(
      "custom_autograd_fn_returns_self",
      &at::native::custom_autograd_fn_returns_self);
  m.impl(
      "custom_autograd_fn_aliasing", &at::native::custom_autograd_fn_aliasing);
}
} // namespace at::openreg

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel_openreg);
REGISTER_PRIVATEUSE1_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_stub_openreg);
REGISTER_PRIVATEUSE1_DISPATCH(
    _fused_sdp_choice_stub,
    &_fused_sdp_choice_openreg);
} // namespace at::native
