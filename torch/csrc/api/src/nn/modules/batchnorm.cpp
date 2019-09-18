#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace torch {
namespace nn {

BatchNormImpl::BatchNormImpl(BatchNormOptions options) : options(options) {
  reset();
}

void BatchNormImpl::reset() {
  if (options.affine_) {
    weight = register_parameter(
        "weight", torch::empty({options.features_}).uniform_());
    bias = register_parameter("bias", torch::zeros({options.features_}));
  }

  if (options.stateful_) {
    running_mean =
        register_buffer("running_mean", torch::zeros({options.features_}));
    running_var =
        register_buffer("running_var", torch::ones({options.features_}));
  }
}

void BatchNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm(features=" << options.features_
         << ", eps=" << options.eps_ << ", momentum=" << options.momentum_
         << ", affine=" << options.affine_ << ", stateful=" << options.stateful_
         << ")";
}

Tensor BatchNormImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      options.stateful_,
      "Calling BatchNorm::forward is only permitted when "
      "the 'stateful' option is true (was false). "
      "Use BatchNorm::pure_forward instead.");
  return pure_forward(input, running_mean, running_var);
}

Tensor BatchNormImpl::pure_forward(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& variance) {
  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    TORCH_CHECK(
        input.numel() / num_channels > 1,
        "BatchNorm expected more than 1 value per channel when training!");
  }

  return torch::batch_norm(
      input,
      weight,
      bias,
      mean,
      variance,
      is_training(),
      options.momentum_,
      options.eps_,
      torch::cuda::cudnn_is_available());
}

} // namespace nn
} // namespace torch
