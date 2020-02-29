#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/modules/batchnorm.h>

#include <torch/cuda.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

BatchNormImpl::BatchNormImpl(const BatchNormOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  TORCH_WARN("torch::nn::BatchNorm module is deprecated and will be removed in 1.5. "
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

// ===========================================================================

template <size_t D, typename Derived> 
void BatchNormImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::BatchNorm" << D << "d("
         << this->options.num_features() << ", "
         << "eps=" << this->options.eps() << ", "
         << "momentum=" << this->options.momentum().value() << ", "
         << "affine=" << this->options.affine() << ", "
         << "track_running_stats=" << this->options.track_running_stats() << ")";
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
