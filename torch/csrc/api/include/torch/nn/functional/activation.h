#pragma once

#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUOptions& options) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkOptions& options) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhOptions& options) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUOptions& options) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

inline Tensor relu(Tensor& input, const ReLUOptions& options) {
  if (options.inplace()) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}

inline Tensor relu6(Tensor& input, const ReLU6Options& options) {
  return hardtanh(input,
    HardtanhOptions().min_val(0).max_val(6).inplace(options.inplace()));
}

inline Tensor rrelu(Tensor& input, const RReLUOptions& options, bool training=false) {
  if (options.inplace()) {
    return torch::rrelu_(input, options.lower(), options.upper(), training);
  } else {
    return torch::rrelu(input, options.lower(), options.upper(), training);
  }
}

inline Tensor multi_head_attention_forward(
  const Tensor& query,
  const Tensor& key,
  const Tensor& value,
  int64_t embed_dim_to_check,
  int64_t num_heads,
  const Tensor& in_proj_weight,
  const Tensor& in_proj_bias,
  const c10::optional<Tensor>& bias_k,
  const c10::optional<Tensor>& bias_v,
  bool add_zero_attn,
  double dropout_p,
  const Tensor& out_proj_weight,
  const Tensor& out_proj_bias,
  bool training = true,
  const c10::optional<Tensor>& key_padding_mask = c10::nullopt,
  bool need_weights = true,
  const c10::optional<Tensor>& attn_mask = c10::nullopt,
  bool use_separate_proj_weight = false,
  const c10::optional<Tensor>& q_proj_weight = c10::nullopt,
  const c10::optional<Tensor>& k_proj_weight = c10::nullopt,
  const c10::optional<Tensor>& v_proj_weight = c10::nullopt,
  const c10::optional<Tensor>& static_k = c10::nullopt,
  const c10::optional<Tensor>& static_v = c10::nullopt) {

  namespace F = torch::nn::functional;

  const bool qkv_same = torch::equal(query, key) && torch::equal(key, value);
  const bool kv_same = torch::equal(key, value);

  const auto query_sizes = query.sizes();
  // const auto& tgt_len = query_sizes[0];
  // const auto& bsz = query_sizes[1];
  const auto& embed_dim = query_sizes[2];
  assert(embed_dim == embed_dim_to_check);
  // assert(query.sizes() == query_sizes);
  assert(key.sizes() == value.sizes());

  const auto head_dim = embed_dim / num_heads;
  assert(head_dim * num_heads == embed_dim);  // "embed_dim must be divisible by num_heads"
  const auto scaling = 1.0 / std::sqrt(head_dim);

  Tensor q, k, v;
  if (!use_separate_proj_weight) {
    if (qkv_same) {
      // self-attention
      const auto chunks = F::linear(query, in_proj_weight, in_proj_bias)
                          .chunk(3, /*dim=*/-1);
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
        // _end = None
        _w = in_proj_weight.slice(/*dim=*/0, _start);
        if (_b.defined()) {
          _b = _b.slice(/*dim=*/0, _start);
        }
        const auto chunks = F::linear(key, _w, _b).chunk(2, /*dim=*/-1);
        k = chunks[0];
        v = chunks[1];
      }
    } else {

    }
  } else {

  }
  return {};
}

} // namespace functional
} // namespace nn
} // namespace torch
