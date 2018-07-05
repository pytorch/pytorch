#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/tensor.h>

#include <ATen/Error.h>

#include <cstddef>
#include <utility>

namespace torch {
namespace nn {
BatchNormOptions::BatchNormOptions(int64_t features) : features_(features) {}

namespace detail {
template <typename Derived>
BatchNormImplBase<Derived>::BatchNormImplBase(BatchNormOptions options)
    : options(std::move(options)) {
  reset();
}

template <typename Derived>
void BatchNormImplBase<Derived>::reset() {
  if (options.affine_) {
    weight = this->register_parameter(
        "weight", torch::empty({options.features_}).uniform_());
    bias = this->register_parameter("bias", torch::zeros({options.features_}));
  }

  if (options.stateful_) {
    running_mean = this->register_buffer(
        "running_mean", torch::zeros({options.features_}));
    running_variance = this->register_buffer(
        "running_variance", torch::ones({options.features_}));
  }
}

template <typename Derived>
Tensor BatchNormImplBase<Derived>::forward(Tensor input) {
  return pure_forward(input, this->running_mean, this->running_variance);
}

template <typename Derived>
Tensor BatchNormImplBase<Derived>::pure_forward(
    Tensor input,
    Tensor mean,
    Tensor variance) {
  check_input_dimensions(input);
  if (this->is_training()) {
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
      this->is_training(),
      options.momentum_,
      options.eps_,
      torch::cuda::cudnn_is_available());
}

template class BatchNormImplBase<BatchNorm1dImpl>;
template class BatchNormImplBase<BatchNorm2dImpl>;
template class BatchNormImplBase<BatchNorm3dImpl>;
} // namespace detail

void BatchNorm1dImpl::check_input_dimensions(Tensor input) const {
  AT_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "Expected 2D or 3D input (got ",
      input.dim(),
      "D input)");
}

void BatchNorm2dImpl::check_input_dimensions(Tensor input) const {
  AT_CHECK(
      input.dim() == 4, "Expected 4D input (got ", input.dim(), "D input)");
}

void BatchNorm3dImpl::check_input_dimensions(Tensor input) const {
  AT_CHECK(
      input.dim() == 5, "Expected 5D input (got ", input.dim(), "D input)");
}

} // namespace nn
} // namespace torch
