#include <torch/nn/modules/batchnorm.h>

namespace torch { namespace nn {
void BatchNorm::initialize_parameters() {
  if (affine_) {
    weight = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "weight");
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "bias");
  }

  if (stateful_) {
    running_mean = Var(DefaultTensor(at::kFloat).zeros({num_features_}), false);
    running_var = Var(DefaultTensor(at::kFloat).ones({num_features_}), false);
  }
}

void BatchNorm::reset_parameters() {
  if (affine_) {
    weight.data().uniform_();
    bias.data().zero_();
  }

  if (stateful_) {
    running_mean.data().zero_();
    running_var.data().fill_(1);
  }
}

variable_list BatchNorm::forward(variable_list inputs) {
  auto& input = inputs[0];
  auto& running_mean = (stateful_ ? this->running_mean : inputs[1]);
  auto& running_var = (stateful_ ? this->running_var : inputs[2]);

  if (train_) {
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
      train_,
      momentum_,
      eps_,
      hasCudnn());

  return variable_list({output});
}
}} // namespace torch::nn
