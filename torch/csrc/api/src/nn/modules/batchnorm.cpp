#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/functional/batchnorm.h>

#include <torch/cuda.h>
#include <torch/types.h>
#include <torch/nn/init.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

BatchNormImpl::BatchNormImpl(const BatchNormOptions& options_) : options(options_) {
  LOG(WARNING) << "torch::nn::BatchNorm module is deprecated."
               << "Use BatchNorm{1,2,3}d instead.";
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

template <size_t D, typename Derived>
BatchNormImplBase<D, Derived>::BatchNormImplBase(const BatchNormOptionsv2<D>& options_)
    : options(options_) {
  if (options.affine()) {
    weight = this->register_parameter("weight", torch::ones({options.num_features()}));
    bias = this->register_parameter("bias", torch::zeros({options.num_features()}));
  }
  if (options.track_running_stats()) {
    running_mean = this->register_buffer("running_mean", torch::zeros({options.num_features()}));
    running_var = this->register_buffer("running_var", torch::ones({options.num_features()}));
    num_batches_tracked = this->register_buffer("num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));
  }
  reset();
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::reset_parameters() {
  reset_running_stats();
  if (options.affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::reset_running_stats() {
  if (options.track_running_stats()) {
    torch::nn::init::zeros_(running_mean);
    torch::nn::init::ones_(running_var);
    torch::nn::init::zeros_(num_batches_tracked);
  }
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::reset() {
  reset_parameters();
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm" << D << "d("
         << "num_features=" << options.num_features() << ", "
         << "eps=" << options.eps() << ", "
         << "momentum=" << options.momentum() << ", "
         << "affine=" << options.affine() << ", "
         << "track_running_stats=" << options.track_running_stats() << ")";
}

Tensor BatchNorm1dImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      input.dim() != 2 && input.dim() !=3,
      "expected 2D or 3D input (got %dD input)", input.dim());         

  if (is_training() && options.track_running_stats()) {
    num_batches_tracked += 1;
  }

  return F::batch_norm1d(
      input,
      running_mean,
      running_var,
      weight,
      bias,
      is_training(),
      options);
}

template class BatchNormImplBase<1, BatchNorm1dImpl>;

} // namespace nn
} // namespace torch
