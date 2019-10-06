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

PReLUImpl::PReLUImpl(const PReLUOptions& options_) : options(options_) {
  reset();
}

Tensor PReLUImpl::forward(const Tensor& input) {
  return F::prelu(input, weight);
}

void PReLUImpl::reset() {
  weight = register_parameter("weight",
    torch::full(options.num_parameters(), options.init()));
}

void PReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::PReLU(num_parameters="
         << options.num_parameters() << ")";
}

// ============================================================================

ReLUImpl::ReLUImpl(const ReLUOptions& options_) : options(options_) {}

Tensor ReLUImpl::forward(Tensor& input) {
  return F::relu(input, options);
}

void ReLUImpl::reset() {}

void ReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU(";
  if (options.inplace()) {
    stream << std::boolalpha  << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

ReLU6Impl::ReLU6Impl(const ReLU6Options& options_) : options(options_) {}

Tensor ReLU6Impl::forward(Tensor& input) {
  return F::relu6(input, options);
}

void ReLU6Impl::reset() {}

void ReLU6Impl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU6(";
  if (options.inplace()) {
    stream << std::boolalpha  << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

RReLUImpl::RReLUImpl(const RReLUOptions& options_) : options(options_) {}

Tensor RReLUImpl::forward(Tensor& input) {
  return F::rrelu(input, options, is_training());
}

void RReLUImpl::reset() {}

void RReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::RReLU(lower=" << options.lower()
         << ", upper=" << options.upper();
  if (options.inplace()) {
    stream << std::boolalpha  << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

MultiheadAttentionImpl::MultiheadAttentionImpl(const MultiheadAttentionOptions& options_)
    : options(options_) {}

Tensor MultiheadAttentionImpl::forward(const Tensor& query, const Tensor& key,
  const Tensor& value, const c10::optional<Tensor>& key_padding_mask,
  bool need_weights, const c10::optional<Tensor>& attn_mask) {
  if (_qkv_same_embed_dim) {
    return F::multi_head_attention_forward(
      query, key, value, options.embed_dim(), options.num_heads(),
      in_proj_weight, in_proj_bias,
      bias_k, bias_v, options.add_zero_attn(),
      options.dropout(), out_proj->weight, out_proj->bias,
      is_training(), key_padding_mask, need_weights, attn_mask);
  } else {
    return F::multi_head_attention_forward(
      query, key, value, options.embed_dim(), options.num_heads(),
      in_proj_weight, in_proj_bias,
      bias_k, bias_v, options.add_zero_attn(),
      options.dropout(), out_proj->weight, out_proj->bias,
      is_training(), key_padding_mask, need_weights,
      attn_mask, /*use_separate_proj_weight=*/true,
      q_proj_weight, k_proj_weight, v_proj_weight);
  }
}

void MultiheadAttentionImpl::reset() {}

void MultiheadAttentionImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MultiheadAttention(todo)";
}

} // namespace nn
} // namespace torch
