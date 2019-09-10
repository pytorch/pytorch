#include <torch/nn/modules/avgpool.h>

#include <torch/expanding_array.h>

namespace torch {
namespace nn {

template <size_t D, typename Derived>
AvgPoolImpl<D, Derived>::AvgPoolImpl(AvgPoolOptions<D> options)
    : options(std::move(options)) {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AvgPool" << D << "d"
         << "(kernel_size=" << options.kernel_size_
         << ", stride=" << options.stride_ << ")";
}

Tensor AvgPool1dImpl::forward(const Tensor& input) {
  return torch::avg_pool1d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.ceil_mode_,
      options.count_include_pad_);
}

Tensor AvgPool2dImpl::forward(const Tensor& input) {
  return torch::avg_pool2d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.ceil_mode_,
      options.count_include_pad_,
      options.divisor_override_);
}

Tensor AvgPool3dImpl::forward(const Tensor& input) {
  return torch::avg_pool3d(
      input,
      options.kernel_size_,
      options.stride_,
      options.padding_,
      options.ceil_mode_,
      options.count_include_pad_,
      options.divisor_override_);
}

template struct AvgPoolOptions<1>;
template class AvgPoolImpl<1, AvgPool1dImpl>;

template struct AvgPoolOptions<2>;
template class AvgPoolImpl<2, AvgPool2dImpl>;

template struct AvgPoolOptions<3>;
template class AvgPoolImpl<3, AvgPool3dImpl>;

} // namespace nn
} // namespace torch
