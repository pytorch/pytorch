#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/functions.h>
#include <torch/tensor.h>

#include <cstdint>

namespace torch {
namespace nn {

BatchNorm::BatchNorm(int64_t features) : features_(features) {}

void BatchNorm::reset() {
  if (affine_) {
    weight_ =
        register_parameter("weight", torch::empty({features_}).uniform_());
    bias_ = register_parameter("bias", torch::zeros({features_}));
  }

  if (stateful_) {
    running_mean_ = register_buffer("running_mean", torch::zeros({features_}));
    running_variance_ =
        register_buffer("running_variance", torch::ones({features_}));
  }
}

std::vector<Variable> BatchNorm::forward(std::vector<Variable> inputs) {
  auto& input = inputs[0];
  auto& running_mean_ = (stateful_ ? this->running_mean_ : inputs[1]);
  auto& running_variance_ = (stateful_ ? this->running_variance_ : inputs[2]);

  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight_,
      bias_,
      running_mean_,
      running_variance_,
      is_training(),
      momentum_,
      eps_,
      torch::cuda::cudnn_is_available());

  return std::vector<Variable>({output});
}
} // namespace nn
} // namespace torch
