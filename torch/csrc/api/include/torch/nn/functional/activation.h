#pragma once

#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>
#include <torch/types.h>
#include <torch/nn/functional/linear.h>
#include <limits>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor selu(Tensor& input, const SELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::selu_(input);
  } else {
    return torch::selu(input);
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkOptions& options = {}) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhOptions& options = {}) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUOptions& options = {}) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor gumbel_softmax(const Tensor& logits, const GumbelSoftmaxOptions& options = {}) {
  auto gumbels = -torch::empty_like(logits).exponential_().log();  // ~Gumbel(0,1)
  gumbels = (logits + gumbels) / options.tau();  // ~Gumbel(logits, tau)
  auto y_soft = gumbels.softmax(options.dim());

  torch::Tensor ret;
  if (options.hard()) {
    // Straight through.
    auto index = std::get<1>(y_soft.max(options.dim(), /*keepdim=*/true));
    auto y_hard = torch::zeros_like(logits).scatter_(options.dim(), index, 1.0);
    ret = y_hard - y_soft.detach() + y_soft;
  } else {
    ret = y_soft;
  }
  return ret;
}

inline Tensor softmax(const Tensor& input, const SoftmaxOptions& options,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}

inline Tensor softmin(const Tensor& input, const SoftminOptions& options,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = (-input).softmax(dim);
  } else {
    ret = (-input).softmax(dim, dtype);
  }

  return ret;
}

inline Tensor log_softmax(const Tensor& input, const LogSoftmaxOptions& options,
                          c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  return ret;
}

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

inline Tensor relu(Tensor& input, const ReLUOptions& options = {}) {
  if (options.inplace()) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}

inline Tensor relu6(Tensor& input, const ReLU6Options& options = {}) {
  return hardtanh(input,
    HardtanhOptions().min_val(0).max_val(6).inplace(options.inplace()));
}

inline Tensor rrelu(Tensor& input, const RReLUOptions& options = {},
                    bool training = false) {
  if (options.inplace()) {
    return torch::rrelu_(input, options.lower(), options.upper(), training);
  } else {
    return torch::rrelu(input, options.lower(), options.upper(), training);
  }
}

inline Tensor celu(Tensor& input, const CELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::celu_(input, options.alpha());
  } else {
    return torch::celu(input, options.alpha());
  }
}

inline Tensor softplus(const Tensor& input,
                       const SoftplusOptions& options = {}) {
  return torch::softplus(input, options.beta(), options.threshold());
}

inline Tensor softshrink(const Tensor& input,
                         const SoftshrinkOptions& options = {}) {
  return torch::softshrink(input, options.lambda());
}

inline Tensor softsign(const Tensor& input) {
  return input / (input.abs() + 1);
}

inline Tensor tanhshrink(const Tensor& input) {
  return input - input.tanh();
}

inline Tensor threshold(Tensor& input, const ThresholdOptions& options) {
  if (options.inplace()) {
    return torch::threshold_(input, options.threshold(), options.value());
  } else {
    return torch::threshold(input, options.threshold(), options.value());
  }
}

inline std::tuple<Tensor, Tensor> multi_head_attention_forward(
  const Tensor& query,
  const Tensor& key,
  const Tensor& value,
  int64_t embed_dim_to_check,
  int64_t num_heads,
  const Tensor& in_proj_weight,
  const Tensor& in_proj_bias,
  const Tensor& bias_k,
  const Tensor& bias_v,
  bool add_zero_attn,
  double dropout_p,
  const Tensor& out_proj_weight,
  const Tensor& out_proj_bias,
  bool training = true,
  const Tensor& key_padding_mask = {},
  bool need_weights = true,
  const Tensor& attn_mask = {},
  bool use_separate_proj_weight = false,
  const Tensor& q_proj_weight = {},
  const Tensor& k_proj_weight = {},
  const Tensor& v_proj_weight = {},
  const Tensor& static_k = {},
  const Tensor& static_v = {}) {
  namespace F = torch::nn::functional;

  const bool qkv_same = torch::equal(query, key) && torch::equal(key, value);
  const bool kv_same = torch::equal(key, value);

  const auto query_sizes = query.sizes();
  const auto& tgt_len = query_sizes[0];
  const auto& bsz = query_sizes[1];
  const auto& embed_dim = query_sizes[2];
  assert(embed_dim == embed_dim_to_check);
  assert(key.sizes() == value.sizes());

  const auto head_dim = embed_dim / num_heads;
  TORCH_CHECK(head_dim * num_heads == embed_dim,
              "embed_dim must be divisible by num_heads");
  const auto scaling = 1 / std::sqrt(head_dim);

  Tensor q, k, v;
  if (!use_separate_proj_weight) {
    if (qkv_same) {
      // self-attention
      const auto chunks =
        F::linear(query, in_proj_weight, in_proj_bias).chunk(3, /*dim=*/-1);
      q = chunks[0];
      k = chunks[1];
      v = chunks[2];
    } else if (kv_same) {
      // encoder-decoder attention
      // This is inline in_proj function with in_proj_weight and in_proj_bias
      auto _b = in_proj_bias;
      auto _start = 0;
      auto _end = embed_dim;
      auto _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      q = F::linear(query, _w, _b);

      if (!key.defined()) {
        assert(!value.defined());
        k = {};
        v = {};
      } else {
        // This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias;
        _start = embed_dim;
        // _end = {};
        _w = in_proj_weight.slice(/*dim=*/0, _start);
        if (_b.defined()) {
          _b = _b.slice(/*dim=*/0, _start);
        }
        const auto chunks = F::linear(key, _w, _b).chunk(2, /*dim=*/-1);
        k = chunks[0];
        v = chunks[1];
      }
    } else {
      auto _b = in_proj_bias;
      auto _start = 0;
      auto _end = embed_dim;
      auto _w = in_proj_weight.slice(0, _start, _end);
      if (_b.defined()) {
        _b = _b.slice(0, _start, _end);
      }
      q = F::linear(query, _w, _b);

      // This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias;
      _start = embed_dim;
      _end = embed_dim * 2;
      _w = in_proj_weight.slice(0, _start, _end);
      if (_b.defined()) {
        _b = _b.slice(0, _start, _end);
      }
      k = F::linear(key, _w, _b);

      // This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias;
      _start = embed_dim * 2;
      // _end = {};
      _w = in_proj_weight.slice(0, _start);
      if (_b.defined()) {
        _b = _b.slice(0, _start);
      }
      v = F::linear(value, _w, _b);
    }
  } else {
    // q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
    auto q_proj_weight_non_opt = q_proj_weight;
    auto sizes = q_proj_weight_non_opt.sizes();
    auto len1 = sizes[0];
    auto len2 = sizes[1];
    TORCH_CHECK(len1 == embed_dim && len2 == query.size(-1));

    // k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
    auto k_proj_weight_non_opt = k_proj_weight;
    sizes = k_proj_weight_non_opt.sizes();
    len1 = sizes[0];
    len2 = sizes[1];
    TORCH_CHECK(len1 == embed_dim && len2 == key.size(-1));

    // v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
    auto v_proj_weight_non_opt = v_proj_weight;
    sizes = v_proj_weight_non_opt.sizes();
    len1 = sizes[0];
    len2 = sizes[1];
    TORCH_CHECK(len1 == embed_dim && len2 == value.size(-1));

    if (in_proj_bias.defined()) {
      q = F::linear(query, q_proj_weight_non_opt, in_proj_bias.slice(0, 0, embed_dim));
      k = F::linear(key, k_proj_weight_non_opt, in_proj_bias.slice(0, embed_dim, (embed_dim * 2)));
      v = F::linear(value, v_proj_weight_non_opt, in_proj_bias.slice(0, (embed_dim * 2)));
    } else {
      q = F::linear(query, q_proj_weight_non_opt, in_proj_bias);
      k = F::linear(key, k_proj_weight_non_opt, in_proj_bias);
      v = F::linear(value, v_proj_weight_non_opt, in_proj_bias);
    }
  }
  q = q * scaling;
  Tensor attn_mask_ = attn_mask;
  Tensor key_padding_mask_ = key_padding_mask;
  if (bias_k.defined() && bias_v.defined()) {
    if (!static_k.defined() && !static_v.defined()) {
      k = torch::cat({k, bias_k.repeat({1, bsz, 1})});
      v = torch::cat({v, bias_v.repeat({1, bsz, 1})});
      if (attn_mask_.defined()) {
        attn_mask_ = torch::cat({
          attn_mask_,
          torch::zeros(
            {attn_mask_.size(0), 1},
            at::TensorOptions(attn_mask_.dtype())
              .device(attn_mask_.device())
          )}, /*dim=*/1
        );
      }
      if (key_padding_mask_.defined()) {
        key_padding_mask_ = torch::cat({
          key_padding_mask_,
          torch::zeros(
            {key_padding_mask_.size(0), 1},
            at::TensorOptions(key_padding_mask_.dtype())
              .device(key_padding_mask_.device())
          )}, /*dim=*/1);
      }
    } else {
      TORCH_CHECK(!static_k.defined(), "bias cannot be added to static key.");
      TORCH_CHECK(!static_v.defined(), "bias cannot be added to static value.");
    }
  } else {
    TORCH_CHECK(!bias_k.defined());
    TORCH_CHECK(!bias_v.defined());
  }
  q = q.contiguous().view({tgt_len, bsz * num_heads, head_dim}).transpose(0, 1);
  if (k.defined()) {
    k = k.contiguous().view({-1, bsz * num_heads, head_dim}).transpose(0, 1);
  }
  if (v.defined()) {
    v = v.contiguous().view({-1, bsz * num_heads, head_dim}).transpose(0, 1);
  }
  if (static_k.defined()) {
    TORCH_CHECK(static_k.size(0) == bsz * num_heads);
    TORCH_CHECK(static_k.size(2) == head_dim);
    k = static_k;
  }
  if (static_v.defined()) {
    TORCH_CHECK(static_v.size(0) == bsz * num_heads);
    TORCH_CHECK(static_v.size(2) == head_dim);
    v = static_v;
  }
  auto src_len = k.size(1);
  if (key_padding_mask_.defined()) {
    TORCH_CHECK(key_padding_mask_.size(0) == bsz);
    TORCH_CHECK(key_padding_mask_.size(1) == src_len);
  }
  if (add_zero_attn) {
    src_len += 1;
    auto k_sizes = k.sizes().vec();
    k_sizes[1] = 1;
    k = torch::cat(
      {
        k,
        torch::zeros(k_sizes, at::TensorOptions(k.dtype()).device(k.device()))
      }, /*dim=*/1);
    auto v_sizes = v.sizes().vec();
    v_sizes[1] = 1;
    v = torch::cat(
      {
        v,
        torch::zeros(v_sizes, at::TensorOptions(v.dtype()).device(v.device()))
      }, /*dim=*/1);
    if (attn_mask_.defined()) {
      attn_mask_ = torch::cat(
        {
          attn_mask_,
          torch::zeros(
            {attn_mask_.size(0), 1},
            at::TensorOptions(attn_mask_.dtype())
              .device(attn_mask_.device()))
        }, /*dim=*/1);
    }
    if (key_padding_mask_.defined()) {
      key_padding_mask_ = torch::cat(
        {
          key_padding_mask_,
          torch::zeros(
            {key_padding_mask_.size(0), 1},
            at::TensorOptions(key_padding_mask_.dtype())
              .device(key_padding_mask_.device()))
        }, /*dim=*/1);
    }
  }
  auto attn_output_weights = torch::bmm(q, k.transpose(1, 2));
  TORCH_CHECK(attn_output_weights.sizes() == IntArrayRef({bsz * num_heads, tgt_len, src_len}));
  if (attn_mask_.defined()) {
    attn_mask_ = attn_mask_.unsqueeze(0);
    attn_output_weights += attn_mask;
  }
  if (key_padding_mask_.defined()) {
    attn_output_weights = attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    attn_output_weights = attn_output_weights.masked_fill(
      key_padding_mask_.unsqueeze(1).unsqueeze(2),
      -std::numeric_limits<float>::infinity()
    );
    attn_output_weights = attn_output_weights.view({bsz * num_heads, tgt_len, src_len});
  }
  attn_output_weights = softmax(attn_output_weights, /*dim=*/1);
  attn_output_weights = dropout(attn_output_weights, /*p=*/dropout_p, /*training=*/training);
  auto attn_output = torch::bmm(attn_output_weights, v);
  TORCH_CHECK(attn_output.sizes() == IntArrayRef({bsz * num_heads, tgt_len, head_dim}));
  attn_output = attn_output.transpose(0, 1).contiguous().view({tgt_len, bsz, embed_dim});
  attn_output = F::linear(attn_output, out_proj_weight, out_proj_bias);
  if (need_weights) {
    // average attention weights over heads
    attn_output_weights = attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    return {attn_output, attn_output_weights.sum(/*dim=*/1) / num_heads};
  } else {
    return {attn_output, {}};
  }
}

inline std::tuple<Tensor, Tensor> multi_head_attention_forward(const MultiheadAttentionForwardOptions& options) {
  return multi_head_attention_forward(
    options.query(),
    options.key(),
    options.value(),
    options.embed_dim_to_check(),
    options.num_heads(),
    options.in_proj_weight(),
    options.in_proj_bias(),
    options.bias_k(),
    options.bias_v(),
    options.add_zero_attn(),
    options.dropout_p(),
    options.out_proj_weight(),
    options.out_proj_bias(),
    options.training(),
    options.key_padding_mask(),
    options.need_weights(),
    options.attn_mask(),
    options.use_separate_proj_weight(),
    options.q_proj_weight(),
    options.k_proj_weight(),
    options.v_proj_weight(),
    options.static_k(),
    options.static_v()
  );
}

} // namespace functional
} // namespace nn
} // namespace torch
