#include <torch/nn/modules/maxpool.h>

#include <torch/expanding_array.h>

namespace torch {
namespace nn {

template <size_t D, typename Derived>
MaxPoolImpl<D, Derived>::MaxPoolImpl(MaxPoolOptions<D> options)
    : options(std::move(options)) {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MaxPool" << D << "d"
         << "(kernel_size=" << options.kernel_size_
         << ", stride=" << options.stride_ << ")";
}

Tensor MaxPool1dImpl::forward(const Tensor& input) {
  return torch::max_pool1d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.dilation_,
      options.ceil_mode_);
}

Tensor MaxPool2dImpl::forward(const Tensor& input) {
  return torch::max_pool2d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.dilation_,
      options.ceil_mode_);
}

Tensor MaxPool3dImpl::forward(const Tensor& input) {
  return torch::max_pool3d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.dilation_,
      options.ceil_mode_);
}

template struct MaxPoolOptions<1>;
template class MaxPoolImpl<1, MaxPool1dImpl>;

template struct MaxPoolOptions<2>;
template class MaxPoolImpl<2, MaxPool2dImpl>;

template struct MaxPoolOptions<3>;
template class MaxPoolImpl<3, MaxPool3dImpl>;

} // namespace nn
} // namespace torch
