#include <torch/nn/modules/batchnorm.h>

namespace torch { namespace nn {

BatchNorm::BatchNorm(uint32_t num_features, bool affine, bool stateful)
    : num_features_(num_features), affine_(affine), stateful_(stateful) {
  if (affine_) {
    weight = add(
        Var(at::CPU(at::kFloat).empty({num_features_}).uniform_()), "weight");
    bias = add(Var(at::CPU(at::kFloat).zeros({num_features_})), "bias");
  }

  if (stateful_) {
    // TODO: Make into buffers instead of parameters
    running_mean = add(
        Var(at::CPU(at::kFloat).zeros({num_features_}), false), "running_mean");
    running_var = add(
        Var(at::CPU(at::kFloat).ones({num_features_}), false), "running_var");
  }
}

variable_list BatchNorm::forward(variable_list inputs) {
  auto& input = inputs[0];
  auto& running_mean = (stateful_ ? this->running_mean : inputs[1]);
  auto& running_var = (stateful_ ? this->running_var : inputs[2]);

  if (is_training()) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      is_training(),
      momentum_,
      eps_,
      hasCudnn());

  return variable_list({output});
}
}} // namespace torch::nn
