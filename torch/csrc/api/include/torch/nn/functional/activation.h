#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor elu(Tensor& input, double alpha = 1.0, bool inplace = false) {
  if (inplace) {
    return torch::elu_(input, alpha);
  } else {
    return torch::elu(input, alpha);
  }
}
} // namespace detail

inline Tensor elu(Tensor& input, ELUFuncOptions options = {}) {
  return detail::elu(input, options.alpha(), options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor selu(Tensor& input, bool inplace = false) {
  if (inplace) {
    return torch::selu_(input);
  } else {
    return torch::selu(input);
  }
}
} // namespace detail

inline Tensor selu(Tensor& input, SELUFuncOptions options = {}) {
  return detail::selu(input, options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor hardshrink(const Tensor& input,
                         double lambda = 0.5) {
  return torch::hardshrink(input, lambda);
}
} // namespace detail

inline Tensor hardshrink(const Tensor& input,
                         HardshrinkFuncOptions options = {}) {
  return detail::hardshrink(input, options.lambda());
}

// ============================================================================

namespace detail {
inline Tensor hardtanh(Tensor& input,
                       double min_val = -1.0,
                       double max_val = 1.0,
                       bool inplace = false) {
  if (inplace) {
    return torch::hardtanh_(input, min_val, max_val);
  } else {
    return torch::hardtanh(input, min_val, max_val);
  }
}
} // namespace detail

inline Tensor hardtanh(Tensor& input, HardtanhFuncOptions options = {}) {
  return detail::hardtanh(input, options.min_val(), options.max_val(), options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor leaky_relu(Tensor& input,
                         double negative_slope = 1e-2,
                         bool inplace = false) {
  if (inplace) {
    return torch::leaky_relu_(input, negative_slope);
  } else {
    return torch::leaky_relu(input, negative_slope);
  }
}
} // namespace detail

inline Tensor leaky_relu(Tensor& input, LeakyReLUFuncOptions options = {}) {
  return detail::leaky_relu(input, options.negative_slope(), options.inplace());
}

// ============================================================================

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

// ============================================================================

namespace detail {
inline Tensor gumbel_softmax(const Tensor& logits,
                             double tau = 1.0,
                             bool hard = false,
                             int dim = -1) {
  auto gumbels = -torch::empty_like(logits).exponential_().log();  // ~Gumbel(0,1)
  gumbels = (logits + gumbels) / tau;  // ~Gumbel(logits, tau)
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

inline Tensor gumbel_softmax(const Tensor& logits, GumbelSoftmaxFuncOptions options = {}) {
  return detail::gumbel_softmax(logits, options.tau(), options.hard(), options.dim());
}

// ============================================================================

namespace detail {
inline Tensor softmax(const Tensor& input, int64_t dim,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail

inline Tensor softmax(const Tensor& input, SoftmaxFuncOptions options) {
  return detail::softmax(input, options.dim(), options.dtype());
}

// ============================================================================

namespace detail {
inline Tensor softmin(const Tensor& input, int64_t dim,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = (-input).softmax(dim);
  } else {
    ret = (-input).softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail

inline Tensor softmin(const Tensor& input, SoftminFuncOptions options) {
  return detail::softmin(input, options.dim(), options.dtype());
}

// ============================================================================

namespace detail {
inline Tensor log_softmax(const Tensor& input, int64_t dim,
                          c10::optional<torch::Dtype> dtype = c10::nullopt) {
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  return ret;
}
} // namespace detail

inline Tensor log_softmax(const Tensor& input, LogSoftmaxFuncOptions options) {
  return detail::log_softmax(input, options.dim(), options.dtype());
}

// ============================================================================

inline Tensor gelu(const Tensor& input) {
  return torch::gelu(input);
}

// ============================================================================

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

// ============================================================================

namespace detail {
inline Tensor relu(Tensor& input, bool inplace = false) {
  if (inplace) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}
} // namespace detail

inline Tensor relu(Tensor& input, ReLUFuncOptions options = {}) {
  return detail::relu(input, options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor relu6(Tensor& input, bool inplace = false) {
  return detail::hardtanh(input, /*min_val=*/0, /*max_val=*/6, /*inplace=*/inplace);
}
} // namespace detail

inline Tensor relu6(Tensor& input, ReLU6FuncOptions options = {}) {
  return detail::relu6(input, options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor rrelu(Tensor& input,
                    double lower = 1.0 / 8.0,
                    double upper = 1.0 / 3.0,
                    bool inplace = false,
                    bool training = false) {
  if (inplace) {
    return torch::rrelu_(input, lower, upper, training);
  } else {
    return torch::rrelu(input, lower, upper, training);
  }
}
} // namespace detail

inline Tensor rrelu(Tensor& input, RReLUFuncOptions options = {}) {
  return detail::rrelu(input, options.lower(), options.upper(), options.inplace(), options.training());
}

// ============================================================================

namespace detail {
inline Tensor celu(Tensor& input,
                   double alpha = 1.0,
                   bool inplace = false) {
  if (inplace) {
    return torch::celu_(input, alpha);
  } else {
    return torch::celu(input, alpha);
  }
}
} // namespace detail

inline Tensor celu(Tensor& input, CELUFuncOptions options = {}) {
  return detail::celu(input, options.alpha(), options.inplace());
}

// ============================================================================

namespace detail {
inline Tensor softplus(const Tensor& input,
                       double beta = 1.0,
                       double threshold = 20.0) {
  return torch::softplus(input, beta, threshold);
}
} // namespace detail

inline Tensor softplus(const Tensor& input,
                       SoftplusFuncOptions options = {}) {
  return detail::softplus(input, options.beta(), options.threshold());
}


// ============================================================================

namespace detail {
inline Tensor softshrink(const Tensor& input,
                         double lambda = 0.5) {
  return torch::softshrink(input, lambda);
}
} // namespace detail

inline Tensor softshrink(const Tensor& input,
                         SoftshrinkFuncOptions options = {}) {
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

namespace detail {
inline Tensor threshold(Tensor& input,
                        double threshold,
                        double value,
                        bool inplace = false) {
  if (inplace) {
    return torch::threshold_(input, threshold, value);
  } else {
    return torch::threshold(input, threshold, value);
  }
}
} // namespace detail

inline Tensor threshold(Tensor& input, ThresholdFuncOptions options) {
  return detail::threshold(input, options.threshold(), options.value(), options.inplace());
}

} // namespace functional
} // namespace nn
} // namespace torch
