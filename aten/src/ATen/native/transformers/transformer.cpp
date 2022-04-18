#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>

#include <torch/library.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

namespace at {

namespace native {

namespace {
Tensor linear_for_ffn(
    const Tensor& bias,
    const Tensor& mat1,
    const Tensor& mat2,
    c10::optional<bool> use_gelu) {
  if (mat1.is_nested()) {
    return NestedTensor_times_Tensor_plus_Tensor_addmm(
        bias, mat1, mat2.t(), 1, 1, use_gelu);
  }

  auto mat1_ = mat1.view({mat1.sizes()[0] * mat1.sizes()[1], mat1.sizes()[2]});
  Tensor result;
  if (use_gelu.has_value()) {
    result = at::_addmm_activation(bias, mat1_, mat2.t(), 1, 1, *use_gelu);
  } else {
    result = at::addmm(bias, mat1_, mat2.t());
  }
  return result.view({mat1.sizes()[0], mat1.sizes()[1], -1});
}

Tensor ffn_cpu(
    const Tensor& input,
    const Tensor& w1,
    const Tensor& b1,
    const Tensor& w2,
    const Tensor& b2,
    bool use_gelu,
    bool add_norm) {
  TORCH_CHECK(add_norm == false, "TODO add_norm to be supported in FFN");
  TORCH_CHECK(input.dim() == 3, "batched input size should be 3");
  TORCH_CHECK(w1.dim() == 2, "2d weights expected");
  TORCH_CHECK(w2.dim() == 2, "2d weights expected");
  Tensor res = linear_for_ffn(b1, input, w1, use_gelu);
  res = linear_for_ffn(b2, res, w2, c10::nullopt);
  return res;
}
} // namespace

Tensor transformer_encoder_layer_forward(
    const Tensor& src,
    const int64_t embed_dim,
    const int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const bool use_gelu,
    const bool norm_first,
    const double layer_norm_eps,
    const Tensor& layer_norm_weight_1,
    const Tensor& layer_norm_bias_1,
    const Tensor& layer_norm_weight_2,
    const Tensor& layer_norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const c10::optional<Tensor>& mask) {
  {
    const Tensor& check_for_empty = src.is_nested() ? get_nested_tensor_impl(src)->get_buffer() : src;
    if (check_for_empty.numel() == 0) {
      return src;
    }
  }
  TORCH_CHECK(!norm_first, "norm_first is not supported yet");
  const bool use_nested_tensor = src.is_nested();
  auto x = std::get<0>(native_multi_head_attention(
      src,
      src,
      src,
      embed_dim,
      num_heads,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      mask,
      false /* need_weights */));
  if (use_nested_tensor) {
    NestedTensor_add_NestedTensor_in_place(x, src);
    x = NestedTensor_layer_norm(
        x, layer_norm_weight_1, layer_norm_bias_1, layer_norm_eps);
  } else {
    x.add_(src);
    x = at::layer_norm(
        x,
        {embed_dim},
        layer_norm_weight_1,
        layer_norm_bias_1,
        layer_norm_eps,
        true);
  }

  auto pre_ffn_res = x;
  x = ffn_cpu(
      x,
      ffn_weight_1,
      ffn_bias_1,
      ffn_weight_2,
      ffn_bias_2,
      use_gelu,
      /* add_norm* */ false);
  if (use_nested_tensor) {
    NestedTensor_add_NestedTensor_in_place(x, pre_ffn_res);
    x = NestedTensor_layer_norm(
        x, layer_norm_weight_2, layer_norm_bias_2, layer_norm_eps);
  } else {
    x.add_(pre_ffn_res);
    x = at::layer_norm(
        x,
        {embed_dim},
        layer_norm_weight_2,
        layer_norm_bias_2,
        layer_norm_eps,
        true);
  }
  return x;
}

} // namespace native
} // namespace at
