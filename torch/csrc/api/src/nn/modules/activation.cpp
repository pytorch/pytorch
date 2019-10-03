#include <torch/nn/modules/activation.h>
#include <torch/nn/functional/activation.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

ELUImpl::ELUImpl(const ELUOptions& options_) : options(options_) {}

Tensor ELUImpl::forward(Tensor& input) {
  return F::elu(input, options);
}

void ELUImpl::reset() {}

void ELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

HardshrinkImpl::HardshrinkImpl(const HardshrinkOptions& options_)
    : options(options_) {}

Tensor HardshrinkImpl::forward(const Tensor& input) {
  return F::hardshrink(input, options);
}

void HardshrinkImpl::reset() {}

void HardshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Hardshrink(" << options.lambda() << ")";
}

// ============================================================================

HardtanhImpl::HardtanhImpl(const HardtanhOptions& options_)
    : options(options_) {
  reset();
}

Tensor HardtanhImpl::forward(Tensor& input) {
  return F::hardtanh(input, options);
}

void HardtanhImpl::reset() {
  TORCH_CHECK(options.max_val() > options.min_val(),
              "max_val must be greater than min_val");
}

void HardtanhImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::Hardtanh(min_val=" << options.min_val()
         << ", max_val=" << options.max_val();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

LeakyReLUImpl::LeakyReLUImpl(const LeakyReLUOptions& options_)
    : options(options_) {}

Tensor LeakyReLUImpl::forward(Tensor& input) {
  return F::leaky_relu(input, options);
}

void LeakyReLUImpl::reset() {}

void LeakyReLUImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LeakyReLU(negative_slope=" << options.negative_slope();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

Tensor LogSigmoidImpl::forward(const Tensor& input) {
  return F::logsigmoid(input);
}

void LogSigmoidImpl::reset() {}

void LogSigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSigmoid()";
}

// ============================================================================

MultiheadAttentionImpl::MultiheadAttentionImpl(const MultiheadAttentionOptions& options_)
    : options(options_) {}

Tensor MultiheadAttentionImpl::forward(const std::map<int, int>& query, int key,
  int value, const c10::optional<Tensor>& key_padding_mask, bool need_weights,
  const c10::optional<int> attn_mask) {
  if (_qkv_same_embed_dim) {
    F::multi_head_attention_forward(
      query, key, value, options.embed_dim, options.num_heads,
      in_proj_weight, in_proj_bias,
      bias_k, bias_v, options.add_zero_attn,
      options.dropout, out_proj.weight, out_proj.bias,
      training,
      key_padding_mask=key_padding_mask, need_weights=need_weights,
      attn_mask=attn_mask);
  } else {
    return F::multi_head_attention_forward(
      query, key, value, options.embed_dim, options.num_heads,
      in_proj_weight, in_proj_bias,
      bias_k, bias_v, options.add_zero_attn,
      options.dropout, out_proj.weight, out_proj.bias,
      training,
      key_padding_mask=key_padding_mask, need_weights=need_weights,
      attn_mask=attn_mask, use_separate_proj_weight=True,
      q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
      v_proj_weight=v_proj_weight);
  }
}

void MultiheadAttentionImpl::reset() {}

void MultiheadAttentionImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiheadAttention(todo)";
}

} // namespace nn
} // namespace torch
