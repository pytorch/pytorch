#include "Extra.h"

namespace at::native {

at::Tensor quantize_per_tensor_openreg(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice_openreg(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
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
_scaled_dot_product_fused_attention_overrideable_openreg(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output =
      at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp =
      at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty(
      {batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
      opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(
      output,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      max_seqlen_q,
      max_seqlen_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward_openreg(
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
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

} // namespace at::native

namespace at::native {

void abs_kernel_openreg(at::TensorIteratorBase& iter) {
  // Abs only have a input tensor and a output tensor.
  auto& output_operand = iter.operand(0);
  auto& input_operand = iter.operand(1);
  auto& output_tensor_base = output_operand.tensor_base();
  auto& input_tensor_base = input_operand.tensor_base();
  TORCH_CHECK(
      !input_operand.original_tensor_base().defined(),
      "input original tensor is defined.");
  TORCH_CHECK(
      !output_operand.original_tensor_base().defined(),
      "output original tensor is defined.");
  // For easy test, only accept contiguous input tensor for calculate.
  auto memory_format = input_tensor_base.suggest_memory_format();
  TORCH_CHECK(
      input_tensor_base.is_contiguous(memory_format),
      "Input tensor need be contiguous.");
  // Add necessary restrictions to ensure the security of the demo.
  TORCH_CHECK(
      input_tensor_base.sizes() == output_tensor_base.sizes(),
      "Intput and output tensor size are not equal.");
  // Common dtype is calculate in TensorIteratorBase.
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float, "Only support float type.")
  // Using for loop for abs calculate.
  auto abs_function =
      [](float* output_ptr, const float* input_ptr, const int64_t NUM) {
        for (int64_t i = 0; i < NUM; ++i) {
          *(output_ptr + i) = std::abs(*(input_ptr + i));
        }
      };
  // To simplify the logic of the test demo code,
  // we only use contiguous tensor to calculate on device side.
  // And using input tensor memory format.
  if (iter.is_contiguous()) {
    // Add for will_resize flag check. You can convert to differernt
    // tensor memory format when will_resize is True.
    // If TensorIteratorConfig resize_outputs_ flag is true, and there are two
    // situations:
    // 1) Out tensor is undefined, and TensorIterator set will_resize to true;
    // 2) Out tensor is defined and tensor size is not equal to input tensor
    // size;
    //    TensorIterator set will_resize to true, and call
    //    set_output_raw_strided to resize output tensor.
    // When output operand will_resize flag is ture, dummy
    // device can convert tensor to dummy device preferred memory format.
    // Here we don't convert tensor memory format, because it will become
    // complex when dummy device want keep same memory format for training
    // network.
    TORCH_CHECK(
        output_operand.will_resize,
        "output operand will_resize flag need be True.");
    abs_function(
        (float*)iter.data_ptr(0), (float*)iter.data_ptr(1), iter.numel());
  } else {
    // Stride copy is not support for foo device, using cpu device instead.
    // For abs op, the last situation is: output tensor is not contiguous with
    // operand will_resize is False.
    TORCH_CHECK(
        !output_operand.will_resize, "output operand will_resize is True.");
    // Get a contiguous tensor with input memory format.
    at::Tensor output = at::empty(
        output_tensor_base.sizes(),
        input_tensor_base.options().memory_format(memory_format));
    // For structured op which inheried from TensorIteratorBase, maybe you need
    // to call set_output_raw_strided function to update output stored in op
    // sturctured. abs op is no need to do this.
    output_operand.exchange_tensor(
        c10::MaybeOwned<at::TensorBase>::owned(std::in_place, output));
    abs_function(
        (float*)output_operand.tensor_base().mutable_data_ptr(),
        (float*)iter.data_ptr(1),
        iter.numel());
    // Copy tensor base to original tensor base, and keep same scalar type and
    // stride with cpu and gpu.
    if (output_operand.original_tensor_base().defined() &&
        !output_operand.original_tensor_base().is_same(
            output_operand.tensor_base())) {
      output_operand.original_tensor().copy_(output_operand.tensor());
      output_operand.restore_original_tensor();
    }
  }
}

void quantize_tensor_per_tensor_affine_stub_openreg(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

} // namespace at::native

namespace at::native {

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

} // namespace at::native
