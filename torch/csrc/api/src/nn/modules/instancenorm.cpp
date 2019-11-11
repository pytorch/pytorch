#include <torch/nn/modules/instancenorm.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
InstanceNormImpl<D, Derived>::InstanceNormImpl(const InstanceNormOptions& options_)
    : BatchNormImplBase<D, Derived>(BatchNormOptions(options_.num_features())),
    options(options_) {}

template <size_t D, typename Derived>
Tensor InstanceNormImpl<D, Derived>::forward(const Tensor& input) {
  _check_input_dim(input);
  return F::instance_norm(input, this->running_mean, this->running_var, this->weight,
      this->bias, this->is_training() || !options.track_running_stats(),
      options.momentum(), options.eps());
}

//template <size_t D, typename Derived>
//void InstanceNormImpl<D, Derived>::reset(){}

void InstanceNorm1dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK( 
      input.dim() == 2, 
      "InstanceNorm1d returns 0-filled tensor to 2D tensor.",                                                                                                             
      "This is because InstanceNorm1d reshapes inputs to",                                                                                                                
      "(1, N * C, ...) from (N, C,...) and this makes",                                                                                                                   
      "variances 0.");
  TORCH_CHECK(
      input.dim() != 3, 
      "expected 3D input (got", input.dim(), "D input)");  
}
/*
void InstanceNorm2dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() != 4, 
      "expected 4D input (got", input.dim(), "D input)");  
}

void InstanceNorm3dImpl::_check_input_dim(const Tensor& input) {
  TORCH_CHECK(
      input.dim() != 5, 
      "expected 5D input (got", input.dim(), "D input)");  
}
*/
template class InstanceNormImpl<1, InstanceNorm1dImpl>;
/*
template class InstanceNormImpl<2, InstanceNorm2dImpl>;
template class InstanceNormImpl<3, InstanceNorm3dImpl>;
*/
} // namespace nn
} // namespace torch
