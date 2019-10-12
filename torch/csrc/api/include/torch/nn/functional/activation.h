#pragma once

#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>
#include <torch/types.h>

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

inline Tensor multi_head_attention_forward(const MultiheadAttentionForwardOptions& options) {
  // // using MultiheadAttentionForwardOptions;

  // namespace F = torch::nn::functional;

  // const bool qkv_same = torch::equal(query, key) && torch::equal(key, value);
  // const bool kv_same = torch::equal(key, value);

  // const auto query_sizes = query.sizes();
  // // const auto& tgt_len = query_sizes[0];
  // // const auto& bsz = query_sizes[1];
  // const auto& embed_dim = query_sizes[2];
  // assert(embed_dim == embed_dim_to_check);
  // // assert(query.sizes() == query_sizes);
  // assert(key.sizes() == value.sizes());

  // const auto head_dim = embed_dim / num_heads;
  // assert(head_dim * num_heads == embed_dim);  // "embed_dim must be divisible by num_heads"
  // const auto scaling = 1.0 / std::sqrt(head_dim);

  // Tensor q, k, v;
  // if (!use_separate_proj_weight) {
  //   if (qkv_same) {
  //     // self-attention
  //     const auto chunks = F::linear(query, in_proj_weight, in_proj_bias)
  //                         .chunk(3, /*dim=*/-1);
  //     q = chunks[0];
  //     k = chunks[1];
  //     v = chunks[2];
  //   } else if (kv_same) {
  //     // encoder-decoder attention
  //     // This is inline in_proj function with in_proj_weight and in_proj_bias
  //     auto _b = in_proj_bias;
  //     auto _start = 0;
  //     auto _end = embed_dim;
  //     auto _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
  //     if (_b.defined()) {
  //       _b = _b.slice(/*dim=*/0, _start, _end);
  //     }
  //     q = F::linear(query, _w, _b);

  //     if (!key.defined()) {
  //       assert(!value.defined());
  //       k = {};
  //       v = {};
  //     } else {
  //       // This is inline in_proj function with in_proj_weight and in_proj_bias
  //       _b = in_proj_bias;
  //       _start = embed_dim;
  //       // _end = None
  //       _w = in_proj_weight.slice(/*dim=*/0, _start);
  //       if (_b.defined()) {
  //         _b = _b.slice(/*dim=*/0, _start);
  //       }
  //       const auto chunks = F::linear(key, _w, _b).chunk(2, /*dim=*/-1);
  //       k = chunks[0];
  //       v = chunks[1];
  //     }
  //   } else {

  //   }
  // } else {

  // }
  return {};
}

} // namespace functional
} // namespace nn
} // namespace torch
