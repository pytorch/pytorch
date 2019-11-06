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

BatchNormImpl::BatchNormImpl(const BatchNormOptions& options_) : options(options_) {
  reset();
}

void BatchNormImpl::reset() {
  if (options.affine()) {
    weight = register_parameter(
        "weight", torch::empty({options.features()}).uniform_());
    bias = register_parameter("bias", torch::zeros({options.features()}));
  }

  if (options.stateful()) {
    running_mean =
        register_buffer("running_mean", torch::zeros({options.features()}));
    running_var =
        register_buffer("running_var", torch::ones({options.features()}));
  }
}

void BatchNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm(features=" << options.features()
         << ", eps=" << options.eps() << ", momentum=" << options.momentum()
         << ", affine=" << options.affine() << ", stateful=" << options.stateful()
         << ")";
}

Tensor BatchNormImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      options.stateful(),
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
      options.momentum(),
      options.eps(),
      torch::cuda::cudnn_is_available());
}

} // namespace nn
} // namespace torch
