#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
BatchNormOptions::BatchNormOptions(int64_t features) : features_(features) {}

BatchNormImpl::BatchNormImpl(BatchNormOptions options)
    : options(std::move(options)) {
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
    running_variance =
        register_buffer("running_variance", torch::ones({options.features_}));
  }
}

Tensor BatchNormImpl::forward(Tensor input) {
  return pure_forward(input, Tensor(), Tensor());
}

Tensor BatchNormImpl::pure_forward(Tensor input, Tensor mean, Tensor variance) {
  auto& running_mean = options.stateful_ ? this->running_mean : mean;
  auto& running_variance =
      options.stateful_ ? this->running_variance : variance;

  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    AT_CHECK(
        input.numel() / num_channels > 1,
        "BatchNorm expected more than 1 value per channel when training!");
  }

  return torch::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_variance,
      is_training(),
      options.momentum_,
      options.eps_,
      torch::cuda::cudnn_is_available());
}

} // namespace nn
} // namespace torch
