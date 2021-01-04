#include <torch/nn/modules/activation.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/init.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

ELUImpl::ELUImpl(const ELUOptions& options_) : options(options_) {}

Tensor ELUImpl::forward(Tensor input) {
  return F::detail::elu(input, options.alpha(), options.inplace());
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

SELUImpl::SELUImpl(const SELUOptions& options_) : options(options_) {}

Tensor SELUImpl::forward(Tensor input) {
  return F::detail::selu(input, options.inplace());
}

void SELUImpl::reset() {}

void SELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SELU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

HardshrinkImpl::HardshrinkImpl(const HardshrinkOptions& options_)
    : options(options_) {}

Tensor HardshrinkImpl::forward(const Tensor& input) {
  return F::detail::hardshrink(input, options.lambda());
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

Tensor HardtanhImpl::forward(Tensor input) {
  return F::detail::hardtanh(input, options.min_val(), options.max_val(), options.inplace());
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

Tensor LeakyReLUImpl::forward(Tensor input) {
  return F::detail::leaky_relu(input, options.negative_slope(), options.inplace());
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

SoftmaxImpl::SoftmaxImpl(const SoftmaxOptions& options_)
    : options(options_) {}

void SoftmaxImpl::reset() {}

void SoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax(dim=" << options.dim() << ")";
}

Tensor SoftmaxImpl::forward(const Tensor& input) {
  return F::detail::softmax(input, options.dim(), c10::nullopt);
}

// ============================================================================

SoftminImpl::SoftminImpl(const SoftminOptions& options_)
    : options(options_) {}

void SoftminImpl::reset() {}

void SoftminImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmin(dim=" << options.dim() << ")";
}

Tensor SoftminImpl::forward(const Tensor& input) {
  return F::detail::softmin(input, options.dim(), c10::nullopt);
}

// ============================================================================

LogSoftmaxImpl::LogSoftmaxImpl(const LogSoftmaxOptions& options_)
    : options(options_) {}

void LogSoftmaxImpl::reset() {}

void LogSoftmaxImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::LogSoftmax(dim=" << options.dim() << ")";
}

Tensor LogSoftmaxImpl::forward(const Tensor& input) {
  return F::detail::log_softmax(input, options.dim(), c10::nullopt);
}

// ============================================================================

void Softmax2dImpl::reset() {}

void Softmax2dImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softmax2d()";
}

Tensor Softmax2dImpl::forward(const Tensor& input) {
  TORCH_CHECK(input.dim() == 4, "Softmax2d requires a 4D tensor as input");
  return F::detail::softmax(input, /*dim=*/1, c10::nullopt);
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

Tensor ReLUImpl::forward(Tensor input) {
  return F::detail::relu(input, options.inplace());
}

void ReLUImpl::reset() {}

void ReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

ReLU6Impl::ReLU6Impl(const ReLU6Options& options_) : options(options_) {}

Tensor ReLU6Impl::forward(Tensor input) {
  return F::detail::relu6(input, options.inplace());
}

void ReLU6Impl::reset() {}

void ReLU6Impl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReLU6(";
  if (options.inplace()) {
    stream << std::boolalpha << "inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

RReLUImpl::RReLUImpl(const RReLUOptions& options_) : options(options_) {}

Tensor RReLUImpl::forward(Tensor input) {
  return F::detail::rrelu(input, options.lower(), options.upper(), is_training(), options.inplace());
}

void RReLUImpl::reset() {}

void RReLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::RReLU(lower=" << options.lower()
         << ", upper=" << options.upper();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

CELUImpl::CELUImpl(const CELUOptions& options_) : options(options_) {}

Tensor CELUImpl::forward(Tensor input) {
  return F::detail::celu(input, options.alpha(), options.inplace());
}

void CELUImpl::reset() {}

void CELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::CELU(alpha=" << options.alpha();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

GLUImpl::GLUImpl(const GLUOptions& options_) : options(options_) {}

Tensor GLUImpl::forward(const Tensor& input) {
  return F::detail::glu(input, options.dim());
}

void GLUImpl::reset() {}

void GLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::GLU(dim=" << options.dim() << ")";
}

// ============================================================================

Tensor GELUImpl::forward(const Tensor& input) {
  return F::gelu(input);
}

void GELUImpl::reset() {}

void GELUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::GELU()";
}

// ============================================================================

Tensor SiLUImpl::forward(const Tensor& input) {
  return F::silu(input);
}

void SiLUImpl::reset() {}

void SiLUImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::SiLU()";
}

// ============================================================================

Tensor SigmoidImpl::forward(const Tensor& input) {
  return torch::sigmoid(input);
}

void SigmoidImpl::reset() {}

void SigmoidImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Sigmoid()";
}

// ============================================================================

SoftplusImpl::SoftplusImpl(const SoftplusOptions& options_)
  : options(options_) {}

Tensor SoftplusImpl::forward(const Tensor& input) {
  return F::detail::softplus(input, options.beta(), options.threshold());
}

void SoftplusImpl::reset() {}

void SoftplusImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softplus(beta=" << options.beta()
         << ", threshold=" << options.threshold() << ")";
}

// ============================================================================

SoftshrinkImpl::SoftshrinkImpl(const SoftshrinkOptions& options_)
    : options(options_) {}

Tensor SoftshrinkImpl::forward(const Tensor& input) {
  return F::detail::softshrink(input, options.lambda());
}

void SoftshrinkImpl::reset() {}

void SoftshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softshrink(" << options.lambda() << ")";
}

// ============================================================================

Tensor SoftsignImpl::forward(const Tensor& input) {
  return F::softsign(input);
}

void SoftsignImpl::reset() {}

void SoftsignImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Softsign()";
}

// ============================================================================

Tensor TanhImpl::forward(const Tensor& input) {
  return torch::tanh(input);
}

void TanhImpl::reset() {}

void TanhImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanh()";
}

// ============================================================================

Tensor TanhshrinkImpl::forward(const Tensor& input) {
  return F::tanhshrink(input);
}

void TanhshrinkImpl::reset() {}

void TanhshrinkImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Tanhshrink()";
}

// ============================================================================

ThresholdImpl::ThresholdImpl(const ThresholdOptions& options_)
    : options(options_) {}

Tensor ThresholdImpl::forward(Tensor input) {
  return F::detail::threshold(input, options.threshold(), options.value(), options.inplace());
}

void ThresholdImpl::reset() {}

void ThresholdImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Threshold(threshold=" << options.threshold()
         << ", value=" << options.value();
  if (options.inplace()) {
    stream << std::boolalpha << ", inplace=" << options.inplace();
  }
  stream << ")";
}

// ============================================================================

MultiheadAttentionImpl::MultiheadAttentionImpl(const MultiheadAttentionOptions& options_)
    : Module("torch::nn::MultiheadAttention"), options(options_) {
  reset();
}

std::tuple<Tensor, Tensor> MultiheadAttentionImpl::forward(
  const Tensor& query, const Tensor& key,
  const Tensor& value, const Tensor& key_padding_mask,
  bool need_weights, const Tensor& attn_mask) {
  if (!_qkv_same_embed_dim) {
    return F::multi_head_attention_forward(
      query, key, value,
      F::MultiheadAttentionForwardFuncOptions(
        /*embed_dim_to_check=*/options.embed_dim(),
        /*num_heads=*/options.num_heads(),
        /*in_proj_weight=*/in_proj_weight,
        /*in_proj_bias=*/in_proj_bias,
        /*bias_k=*/bias_k,
        /*bias_v=*/bias_v,
        /*add_zero_attn=*/options.add_zero_attn(),
        /*dropout_p=*/options.dropout(),
        /*out_proj_weight=*/out_proj->weight,
        /*out_proj_bias=*/out_proj->bias
      ).training(is_training())
       .key_padding_mask(key_padding_mask)
       .need_weights(need_weights)
       .attn_mask(attn_mask)
       .use_separate_proj_weight(true)
       .q_proj_weight(q_proj_weight)
       .k_proj_weight(k_proj_weight)
       .v_proj_weight(v_proj_weight)
    );
  } else {
    return F::multi_head_attention_forward(
      query, key, value,
      F::MultiheadAttentionForwardFuncOptions(
        /*embed_dim_to_check=*/options.embed_dim(),
        /*num_heads=*/options.num_heads(),
        /*in_proj_weight=*/in_proj_weight,
        /*in_proj_bias=*/in_proj_bias,
        /*bias_k=*/bias_k,
        /*bias_v=*/bias_v,
        /*add_zero_attn=*/options.add_zero_attn(),
        /*dropout_p=*/options.dropout(),
        /*out_proj_weight=*/out_proj->weight,
        /*out_proj_bias=*/out_proj->bias
      ).training(is_training())
       .key_padding_mask(key_padding_mask)
       .need_weights(need_weights)
       .attn_mask(attn_mask)
    );
  }
}

void MultiheadAttentionImpl::reset() {
  _qkv_same_embed_dim = options.kdim() == options.embed_dim() &&
                        options.vdim() == options.embed_dim();
  head_dim = options.embed_dim() / options.num_heads();
  TORCH_CHECK(head_dim * options.num_heads() == options.embed_dim(),
              "embed_dim must be divisible by num_heads");
  if (!_qkv_same_embed_dim) {
    q_proj_weight = register_parameter(
      "q_proj_weight", torch::empty({options.embed_dim(), options.embed_dim()}));
    k_proj_weight = register_parameter(
      "k_proj_weight", torch::empty({options.embed_dim(), options.kdim()}));
    v_proj_weight = register_parameter(
      "v_proj_weight", torch::empty({options.embed_dim(), options.vdim()}));
    register_parameter("in_proj_weight", {}, /*requires_grad=*/false);
  } else {
    in_proj_weight = register_parameter(
      "in_proj_weight", torch::empty({3 * options.embed_dim(), options.embed_dim()}));
    register_parameter("q_proj_weight", {}, /*requires_grad=*/false);
    register_parameter("k_proj_weight", {}, /*requires_grad=*/false);
    register_parameter("v_proj_weight", {}, /*requires_grad=*/false);
  }
  if (options.bias()) {
    in_proj_bias = register_parameter(
      "in_proj_bias", torch::empty(3 * options.embed_dim()));
  } else {
    register_parameter("in_proj_bias", {}, /*requires_grad=*/false);
  }
  out_proj = register_module(
    "out_proj",
    Linear(LinearOptions(options.embed_dim(),
      options.embed_dim()).bias(options.bias()))
  );
  if (options.add_bias_kv()) {
    bias_k = register_parameter("bias_k", torch::empty({1, 1, options.embed_dim()}));
    bias_v = register_parameter("bias_v", torch::empty({1, 1, options.embed_dim()}));
  } else {
    bias_k = {};
    bias_v = {};
  }
  _reset_parameters();
}

void MultiheadAttentionImpl::_reset_parameters() {
  using namespace torch::nn::init;
  if (_qkv_same_embed_dim) {
    xavier_uniform_(in_proj_weight);
  } else {
    xavier_uniform_(q_proj_weight);
    xavier_uniform_(k_proj_weight);
    xavier_uniform_(v_proj_weight);
  }
  if (in_proj_bias.defined()) {
    constant_(in_proj_bias, 0.);
    constant_(out_proj->bias, 0.);
  }
  if (bias_k.defined()) {
    xavier_normal_(bias_k);
  }
  if (bias_v.defined()) {
    xavier_normal_(bias_v);
  }
}

} // namespace nn
} // namespace torch
