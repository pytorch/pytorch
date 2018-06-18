#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/functions.h>
#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace nn {
BatchNormOptions::BatchNormOptions(int64_t features) : features_(features) {}

BatchNormImpl::BatchNormImpl(BatchNormOptions options)
    : options_(std::move(options)) {
  reset();
}

void BatchNormImpl::reset() {
  if (options_.affine_) {
    weight_ = register_parameter(
        "weight", torch::empty({options_.features_}).uniform_());
    bias_ = register_parameter("bias", torch::zeros({options_.features_}));
  }

  if (options_.stateful_) {
    running_mean_ =
        register_buffer("running_mean", torch::zeros({options_.features_}));
    running_variance_ =
        register_buffer("running_variance", torch::ones({options_.features_}));
  }
}

std::vector<Variable> BatchNormImpl::forward(std::vector<Variable> inputs) {
  auto& input = inputs[0];
  auto& running_mean_ = (options_.stateful_ ? this->running_mean_ : inputs[1]);
  auto& running_variance_ =
      (options_.stateful_ ? this->running_variance_ : inputs[2]);

  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    AT_CHECK(
        input.numel() / num_channels > 1,
        "BatchNorm expected more than 1 value per channel when training!");
  }

  auto output = at::batch_norm(
      input,
      weight_,
      bias_,
      running_mean_,
      running_variance_,
      is_training(),
      options_.momentum_,
      options_.eps_,
      torch::cuda::cudnn_is_available());

  return std::vector<Variable>({output});
}

const BatchNormOptions& BatchNormImpl::options() const noexcept {
  return options_;
}

} // namespace nn
} // namespace torch
