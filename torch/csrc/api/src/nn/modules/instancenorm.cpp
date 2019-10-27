#include <torch/nn/modules/instancenorm.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
InstanceNormImpl<D, Derived>::InstanceNormImpl(const BatchNormOptions& options_){}

template <size_t D, typename Derived>
Tensor InstanceNormImpl<D, Derived>::forward(const Tensor& input) {
  _check_input_dim(input);
  return F::instance_norm(input, running_mean, running_var, weight, bias, this->is_training() ||
      !options.track_running_stats(), options.momentum(), options.eps());
}

void InstanceNorm1dImpl::_check_input_dim(const Tensor& input) {
  if (input.dim() == 2) {
    //TODO: raise valueError
  } else if (input.dim() != 3) {
    //TODO: raise valueError
  }
}

template class InstanceNormImpl<1, InstanceNorm1dImpl>;
template class InstanceNormImpl<2, InstanceNorm2dImpl>;
template class InstanceNormImpl<3, InstanceNorm3dImpl>;

} // namespace nn
} // namespace torch
