#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/modules/batchnorm.h>

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
  TORCH_WARN("torch::nn::BatchNorm module is deprecated."
             "Use BatchNorm{1,2,3}d instead.");
  reset();
}

void BatchNormImpl::reset() {
  if (options.affine()) {
    weight = register_parameter(
        "weight", torch::empty({options.num_features()}).uniform_());
    bias = register_parameter("bias", torch::zeros({options.num_features()}));
  }

  if (options.track_running_stats()) {
    running_mean =
        register_buffer("running_mean", torch::zeros({options.num_features()}));
    running_var =
        register_buffer("running_var", torch::ones({options.num_features()}));
  }
}

void BatchNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm(num_features=" << options.num_features()
         << ", eps=" << options.eps() << ", momentum=" << options.momentum().value()
         << ", affine=" << options.affine() << ", track_running_stats=" << options.track_running_stats()
         << ")";
}

Tensor BatchNormImpl::forward(const Tensor& input) {
  TORCH_CHECK(
      options.track_running_stats(),
      "Calling BatchNorm::forward is only permitted when "
      "the 'track_running_stats' option is true (was false). "
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
      options.momentum().value(),
      options.eps(),
      torch::cuda::cudnn_is_available());
}

template <size_t D, typename Derived>
BatchNormImplBase<D, Derived>::BatchNormImplBase(const BatchNormOptions& options_)
    : options(options_) {
  reset();
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::reset_running_stats() {
  if (options.track_running_stats()) {
    running_mean.zero_();
    running_var.fill_(1);
    num_batches_tracked.zero_();
  }
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::reset() {
  if (options.affine()) {
    weight = this->register_parameter("weight", torch::empty({options.num_features()}));
    bias = this->register_parameter("bias", torch::empty({options.num_features()}));
  } else {
    weight = this->register_parameter("weight", Tensor());
    bias = this->register_parameter("bias", Tensor());
  }
  if (options.track_running_stats()) {
    running_mean = this->register_buffer("running_mean", torch::zeros({options.num_features()}));
    running_var = this->register_buffer("running_var", torch::ones({options.num_features()}));
    num_batches_tracked = this->register_buffer("num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));
  } else {
    running_mean = this->register_buffer("running_mean", Tensor());
    running_var = this->register_buffer("running_var", Tensor());
    num_batches_tracked = this->register_buffer("num_batches_tracked", Tensor());
  }

  reset_running_stats();
  if (options.affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

template <size_t D, typename Derived>
void BatchNormImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm" << D << "d("
         << options.num_features() << ", "
         << "eps=" << options.eps() << ", "
         << "momentum=" << options.momentum().value() << ", "
         << "affine=" << options.affine() << ", "
         << "track_running_stats=" << options.track_running_stats() << ")";
}

template <size_t D, typename Derived>
Tensor BatchNormImplBase<D, Derived>::forward(const Tensor& input) {
  _check_input_dim(input);

  double exponential_average_factor;
  if (options.momentum() == c10::nullopt) {
    exponential_average_factor = 0.0;
  } else {
    exponential_average_factor = options.momentum().value();
  }

  if (this->is_training() && options.track_running_stats()) {
    if (num_batches_tracked.defined()) {
      num_batches_tracked += 1;
      if (options.momentum() == c10::nullopt) {  // use cumulative moving average
        exponential_average_factor = 1.0 / num_batches_tracked.item<double>();
      } else {  // use exponential moving average
        exponential_average_factor = options.momentum().value();
      }
    }
  }

  return F::batch_norm(
      input,
      running_mean,
      running_var,
      F::BatchNormFuncOptions().weight(weight).bias(bias).momentum(exponential_average_factor).eps(options.eps()),
      this->is_training() || !options.track_running_stats());
}

void BatchNorm1dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "expected 2D or 3D input (got ", input.dim(), "D input)");
}

void BatchNorm2dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 4,
      "expected 4D input (got ", input.dim(), "D input)");
}

void BatchNorm3dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() == 5,
      "expected 5D input (got ", input.dim(), "D input)");
}

template class BatchNormImplBase<1, BatchNorm1dImpl>;
template class BatchNormImplBase<2, BatchNorm2dImpl>;
template class BatchNormImplBase<3, BatchNorm3dImpl>;

} // namespace nn
} // namespace torch
