#pragma once

#include <torch/nn/options/instancenorm.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor instance_norm(const Tensor& input, const Tensor& running_mean,
    const Tensor& running_var, const Tensor& weight, const Tensor& bias,
    bool use_input_stats, double momentum, double eps) {

  return torch::instance_norm(
    input, weight, bias, running_mean, running_var,
    use_input_stats, momentum, eps, at::globalContext().userEnabledCuDNN()
  );
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.instance_norm
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::InstanceNormFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::instance_norm(input, F::InstanceNormFuncOptions().running_mean(mean).running_var(variance).weight(weight).bias(bias).momentum(0.1).eps(1e-5));
/// ```
inline Tensor instance_norm(const Tensor& input, const InstanceNormFuncOptions& options = {}) {
  return detail::instance_norm(
    input, options.running_mean(),
    options.running_var(), options.weight(), options.bias(),
    options.use_input_stats(), options.momentum(), options.eps());
}

} // namespace functional
} // namespace nn
} // namespace torch
