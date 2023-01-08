#pragma once

#include <ATen/Dispatch.h>
#include <torch/nn/functional/dropout.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/dropout.h>
#include <torch/nn/options/linear.h>
#include <torch/types.h>
#include <limits>
#include <utility>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor elu(Tensor input, double alpha, bool inplace) {
  if (inplace) {
    return torch::elu_(input, alpha);
  } else {
    return torch::elu(input, alpha);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.elu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ELUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::elu(x, F::ELUFuncOptions().alpha(0.42).inplace(true));
/// ```
inline Tensor elu(Tensor input, const ELUFuncOptions& options = {}) {
  return detail::elu(std::move(input), options.alpha(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor selu(Tensor input, bool inplace) {
  if (inplace) {
    return torch::selu_(input);
  } else {
    return torch::selu(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.selu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SELUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::selu(input, F::SELUFuncOptions(false));
/// ```
inline Tensor selu(Tensor input, const SELUFuncOptions& options = {}) {
  return detail::selu(std::move(input), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor hardshrink(const Tensor& input, double lambda) {
  return torch::hardshrink(input, lambda);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.hardshrink
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::HardshrinkFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hardshrink(x, F::HardshrinkFuncOptions().lambda(0.42));
/// ```
inline Tensor hardshrink(
    const Tensor& input,
    const HardshrinkFuncOptions& options = {}) {
  return detail::hardshrink(input, options.lambda());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor hardtanh(
    Tensor input,
    double min_val,
    double max_val,
    bool inplace) {
  if (inplace) {
    return torch::hardtanh_(input, min_val, max_val);
  } else {
    return torch::hardtanh(input, min_val, max_val);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.hardtanh
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::HardtanhFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::hardtanh(x,
/// F::HardtanhFuncOptions().min_val(-1.0).max_val(1.0).inplace(true));
/// ```
inline Tensor hardtanh(Tensor input, const HardtanhFuncOptions& options = {}) {
  return detail::hardtanh(
      std::move(input),
      options.min_val(),
      options.max_val(),
      options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor leaky_relu(Tensor input, double negative_slope, bool inplace) {
  if (inplace) {
    return torch::leaky_relu_(input, negative_slope);
  } else {
    return torch::leaky_relu(input, negative_slope);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.leaky_relu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LeakyReLUFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::leaky_relu(x,
/// F::LeakyReLUFuncOptions().negative_slope(0.42).inplace(true));
/// ```
inline Tensor leaky_relu(
    Tensor input,
    const LeakyReLUFuncOptions& options = {}) {
  return detail::leaky_relu(
      std::move(input), options.negative_slope(), options.inplace());
}

// ============================================================================

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor gumbel_softmax(
    const Tensor& logits,
    double tau,
    bool hard,
    int dim) {
  auto gumbels =
      -torch::empty_like(logits).exponential_().log(); // ~Gumbel(0,1)
  gumbels = (logits + gumbels) / tau; // ~Gumbel(logits, tau)
  auto y_soft = gumbels.softmax(dim);

  torch::Tensor ret;
  if (hard) {
    // Straight through.
    auto index = std::get<1>(y_soft.max(dim, /*keepdim=*/true));
    auto y_hard = torch::zeros_like(logits).scatter_(dim, index, 1.0);
    ret = y_hard - y_soft.detach() + y_soft;
  } else {
    ret = y_soft;
  }
  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.gumbel_softmax
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GumbelSoftmaxFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::gumbel_softmax(logits, F::GumbelSoftmaxFuncOptions().hard(true).dim(-1));
/// ```
inline Tensor gumbel_softmax(
    const Tensor& logits,
    const GumbelSoftmaxFuncOptions& options = {}) {
  return detail::gumbel_softmax(
      logits, options.tau(), options.hard(), options.dim());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor softmax(
    const Tensor& input,
    int64_t dim,
    c10::optional<torch::Dtype> dtype) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.softmax
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SoftmaxFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softmax(input, F::SoftmaxFuncOptions(1));
/// ```
inline Tensor softmax(const Tensor& input, const SoftmaxFuncOptions& options) {
  return detail::softmax(input, options.dim(), options.dtype());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor softmin(
    const Tensor& input,
    int64_t dim,
    c10::optional<torch::Dtype> dtype) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = (-input).softmax(dim);
  } else {
    ret = (-input).softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.softmin
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SoftminFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softmin(input, F::SoftminFuncOptions(1));
/// ```
inline Tensor softmin(const Tensor& input, const SoftminFuncOptions& options) {
  return detail::softmin(input, options.dim(), options.dtype());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor log_softmax(
    const Tensor& input,
    int64_t dim,
    c10::optional<torch::Dtype> dtype) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.log_softmax
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LogSoftmaxFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::log_softmax(input, LogSoftmaxFuncOptions(1));
/// ```
inline Tensor log_softmax(
    const Tensor& input,
    const LogSoftmaxFuncOptions& options) {
  return detail::log_softmax(input, options.dim(), options.dtype());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor glu(const Tensor& input, int64_t dim) {
  TORCH_CHECK(
      input.dim() != 0,
      "glu does not suppport scalars because halving size must be even");
  return torch::glu(input, dim);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GLUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::glu(input, GLUFuncOptions(1));
/// ```
inline Tensor glu(const Tensor& input, const GLUFuncOptions& options = {}) {
  return detail::glu(input, options.dim());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor gelu(const Tensor& input, string approximate) {
  return torch::gelu(input, approximate);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

inline Tensor gelu(const Tensor& input, const GELUFuncOptions& options = {}) {
  return detail::gelu(input, options.approximate());
}

// ============================================================================

inline Tensor silu(const Tensor& input) {
  return torch::silu(input);
}

// ============================================================================

inline Tensor mish(const Tensor& input) {
  return torch::mish(input);
}

// ============================================================================

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor relu(Tensor input, bool inplace) {
  if (inplace) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.relu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ReLUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu(x, F::ReLUFuncOptions().inplace(true));
/// ```
inline Tensor relu(Tensor input, const ReLUFuncOptions& options = {}) {
  return detail::relu(std::move(input), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor relu6(Tensor input, bool inplace) {
  if (inplace) {
    return torch::relu6_(input);
  } else {
    return torch::relu6(input);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.relu6
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ReLU6FuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::relu6(x, F::ReLU6FuncOptions().inplace(true));
/// ```
inline Tensor relu6(Tensor input, const ReLU6FuncOptions& options = {}) {
  return detail::relu6(std::move(input), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor rrelu(
    Tensor input,
    double lower,
    double upper,
    bool training,
    bool inplace) {
  if (inplace) {
    return torch::rrelu_(input, lower, upper, training);
  } else {
    return torch::rrelu(input, lower, upper, training);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.rrelu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::RReLUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::rrelu(x, F::RReLUFuncOptions().lower(0.1).upper(0.4).inplace(true));
/// ```
inline Tensor rrelu(Tensor input, const RReLUFuncOptions& options = {}) {
  return detail::rrelu(
      std::move(input),
      options.lower(),
      options.upper(),
      options.training(),
      options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor celu(Tensor input, double alpha, bool inplace) {
  if (inplace) {
    return torch::celu_(input, alpha);
  } else {
    return torch::celu(input, alpha);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.celu
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::CELUFuncOptions` class to
/// learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::celu(x, F::CELUFuncOptions().alpha(0.42).inplace(true));
/// ```
inline Tensor celu(Tensor input, const CELUFuncOptions& options = {}) {
  return detail::celu(std::move(input), options.alpha(), options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor softplus(const Tensor& input, double beta, double threshold) {
  return torch::softplus(input, beta, threshold);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.softplus
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SoftplusFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softplus(x, F::SoftplusFuncOptions().beta(0.5).threshold(3.0));
/// ```
inline Tensor softplus(
    const Tensor& input,
    const SoftplusFuncOptions& options = {}) {
  return detail::softplus(input, options.beta(), options.threshold());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor softshrink(const Tensor& input, double lambda) {
  return torch::softshrink(input, lambda);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.softshrink
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::SoftshrinkFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::softshrink(x, F::SoftshrinkFuncOptions(0.42));
/// ```
inline Tensor softshrink(
    const Tensor& input,
    const SoftshrinkFuncOptions& options = {}) {
  return detail::softshrink(input, options.lambda());
}

// ============================================================================

inline Tensor softsign(const Tensor& input) {
  return input / (input.abs() + 1);
}

// ============================================================================

inline Tensor tanhshrink(const Tensor& input) {
  return input - input.tanh();
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor threshold(
    Tensor input,
    double threshold,
    double value,
    bool inplace) {
  if (inplace) {
    return torch::threshold_(input, threshold, value);
  } else {
    return torch::threshold(input, threshold, value);
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.threshold
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ThresholdFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::threshold(x, F::ThresholdFuncOptions(0.5, 0.5).inplace(true));
/// ```
inline Tensor threshold(Tensor input, const ThresholdFuncOptions& options) {
  return detail::threshold(
      std::move(input),
      options.threshold(),
      options.value(),
      options.inplace());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
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
    const Tensor& static_v = {},
    bool average_attn_weights = true) {
  namespace F = torch::nn::functional;

  const auto query_sizes = query.sizes();
  const auto& tgt_len = query_sizes[0];
  const auto& bsz = query_sizes[1];
  const auto& embed_dim = query_sizes[2];
  TORCH_INTERNAL_ASSERT(embed_dim == embed_dim_to_check);
  TORCH_INTERNAL_ASSERT(key.sizes() == value.sizes());

  const auto head_dim = embed_dim / num_heads;
  TORCH_CHECK(
      head_dim * num_heads == embed_dim,
      "embed_dim must be divisible by num_heads");
  const auto scaling = 1 / std::sqrt(head_dim);

  Tensor q, k, v;
  if (!use_separate_proj_weight) {
    if (torch::equal(query, key) && torch::equal(key, value)) {
      // self-attention
      const auto chunks =
          F::linear(query, in_proj_weight, in_proj_bias).chunk(3, /*dim=*/-1);
      q = chunks[0];
      k = chunks[1];
      v = chunks[2];
    } else if (torch::equal(key, value)) {
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
        TORCH_INTERNAL_ASSERT(!value.defined());
        k.reset();
        v.reset();
      } else {
        // This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias;
        _start = embed_dim;
        _w = in_proj_weight.slice(/*dim=*/0, _start);
        if (_b.defined()) {
          _b = _b.slice(/*dim=*/0, _start);
        }
        const auto chunks = F::linear(key, _w, _b).chunk(2, /*dim=*/-1);
        k = chunks[0];
        v = chunks[1];
      }
    } else {
      // This is inline in_proj function with in_proj_weight and in_proj_bias
      auto _b = in_proj_bias;
      auto _start = 0;
      auto _end = embed_dim;
      auto _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      q = F::linear(query, _w, _b);

      // This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias;
      _start = embed_dim;
      _end = embed_dim * 2;
      _w = in_proj_weight.slice(/*dim=*/0, _start, _end);
      if (_b.defined()) {
        _b = _b.slice(/*dim=*/0, _start, _end);
      }
      k = F::linear(key, _w, _b);

      // This is inline in_proj function with in_proj_weight and in_proj_bias
      _b = in_proj_bias;
      _start = embed_dim * 2;
      _w = in_proj_weight.slice(/*dim=*/0, _start);
      if (_b.defined()) {
        _b = _b.slice(0, _start);
      }
      v = F::linear(value, _w, _b);
    }
  } else {
    const auto& q_proj_weight_non_opt = q_proj_weight;
    {
      const auto sizes = q_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == query.size(-1));
    }

    const auto& k_proj_weight_non_opt = k_proj_weight;
    {
      const auto sizes = k_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == key.size(-1));
    }

    const auto& v_proj_weight_non_opt = v_proj_weight;
    {
      const auto sizes = v_proj_weight_non_opt.sizes();
      const auto len1 = sizes[0];
      const auto len2 = sizes[1];
      TORCH_CHECK(len1 == embed_dim && len2 == value.size(-1));
    }

    if (in_proj_bias.defined()) {
      q = F::linear(
          query,
          q_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, 0, embed_dim));
      k = F::linear(
          key,
          k_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, embed_dim, (embed_dim * 2)));
      v = F::linear(
          value,
          v_proj_weight_non_opt,
          in_proj_bias.slice(/*dim=*/0, (embed_dim * 2)));
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
        attn_mask_ = torch::cat(
            {attn_mask_,
             torch::zeros(
                 {attn_mask_.size(0), 1},
                 at::TensorOptions(attn_mask_.dtype())
                     .device(attn_mask_.device()))},
            /*dim=*/1);
      }
      if (key_padding_mask_.defined()) {
        key_padding_mask_ = torch::cat(
            {key_padding_mask_,
             torch::zeros(
                 {key_padding_mask_.size(0), 1},
                 at::TensorOptions(key_padding_mask_.dtype())
                     .device(key_padding_mask_.device()))},
            /*dim=*/1);
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
        {k,
         torch::zeros(
             k_sizes, at::TensorOptions(k.dtype()).device(k.device()))},
        /*dim=*/1);
    auto v_sizes = v.sizes().vec();
    v_sizes[1] = 1;
    v = torch::cat(
        {v,
         torch::zeros(
             v_sizes, at::TensorOptions(v.dtype()).device(v.device()))},
        /*dim=*/1);
    if (attn_mask_.defined()) {
      attn_mask_ = torch::cat(
          {attn_mask_,
           torch::zeros(
               {attn_mask_.size(0), 1},
               at::TensorOptions(attn_mask_.dtype())
                   .device(attn_mask_.device()))},
          /*dim=*/1);
    }
    if (key_padding_mask_.defined()) {
      key_padding_mask_ = torch::cat(
          {key_padding_mask_,
           torch::zeros(
               {key_padding_mask_.size(0), 1},
               at::TensorOptions(key_padding_mask_.dtype())
                   .device(key_padding_mask_.device()))},
          /*dim=*/1);
    }
  }
  auto attn_output_weights = torch::bmm(q, k.transpose(1, 2));
  TORCH_CHECK(
      attn_output_weights.sizes() ==
      IntArrayRef({bsz * num_heads, tgt_len, src_len}));
  if (attn_mask_.defined()) {
    attn_mask_ = attn_mask_.unsqueeze(0);
    attn_output_weights += attn_mask_;
  }
  if (key_padding_mask_.defined()) {
    attn_output_weights =
        attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    attn_output_weights = AT_DISPATCH_FLOATING_TYPES(
        attn_output_weights.scalar_type(),
        "attn_output_weights.masked_fill",
        [&]() {
          return attn_output_weights.masked_fill(
              key_padding_mask_.unsqueeze(1).unsqueeze(2),
              -std::numeric_limits<scalar_t>::infinity());
        });
    attn_output_weights =
        attn_output_weights.view({bsz * num_heads, tgt_len, src_len});
  }
  // NOLINTNEXTLINE(bugprone-argument-comment)
  attn_output_weights = F::softmax(attn_output_weights, /*dim=*/-1);
  attn_output_weights = F::dropout(
      attn_output_weights,
      F::DropoutFuncOptions().p(dropout_p).training(training));
  auto attn_output = torch::bmm(attn_output_weights, v);
  TORCH_CHECK(
      attn_output.sizes() == IntArrayRef({bsz * num_heads, tgt_len, head_dim}));
  attn_output =
      attn_output.transpose(0, 1).contiguous().view({tgt_len, bsz, embed_dim});
  attn_output = F::linear(attn_output, out_proj_weight, out_proj_bias);
  if (need_weights) {
    attn_output_weights =
        attn_output_weights.view({bsz, num_heads, tgt_len, src_len});
    if (average_attn_weights) {
      // average attention weights over heads
      attn_output_weights = attn_output_weights.sum(/*dim=*/1) / num_heads;
    }
    return std::make_tuple(attn_output, attn_output_weights);
  } else {
    return std::make_tuple(attn_output, Tensor());
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

inline std::tuple<Tensor, Tensor> multi_head_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const MultiheadAttentionForwardFuncOptions& options) {
  return detail::multi_head_attention_forward(
      query,
      key,
      value,
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
      options.static_v(),
      options.average_attn_weights());
}

} // namespace functional
} // namespace nn
} // namespace torch
