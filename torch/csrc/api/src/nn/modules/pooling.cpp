#include <torch/nn/modules/pooling.h>

#include <torch/expanding_array.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

template <size_t D, typename Derived>
AvgPoolImpl<D, Derived>::AvgPoolImpl(const AvgPoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void AvgPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AvgPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ")";
}

Tensor AvgPool1dImpl::forward(const Tensor& input) {
  return F::avg_pool1d(input, options);
}

Tensor AvgPool2dImpl::forward(const Tensor& input) {
  return F::avg_pool2d(input, options);
}

Tensor AvgPool3dImpl::forward(const Tensor& input) {
  return F::avg_pool3d(input, options);
}

template class AvgPoolImpl<1, AvgPool1dImpl>;
template class AvgPoolImpl<2, AvgPool2dImpl>;
template class AvgPoolImpl<3, AvgPool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
MaxPoolImpl<D, Derived>::MaxPoolImpl(const MaxPoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void MaxPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::MaxPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ")";
}

Tensor MaxPool1dImpl::forward(const Tensor& input) {
  return F::max_pool1d(input, options);
}

Tensor MaxPool2dImpl::forward(const Tensor& input) {
  return F::max_pool2d(input, options);
}

Tensor MaxPool3dImpl::forward(const Tensor& input) {
  return F::max_pool3d(input, options);
}

template class MaxPoolImpl<1, MaxPool1dImpl>;
template class MaxPoolImpl<2, MaxPool2dImpl>;
template class MaxPoolImpl<3, MaxPool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
AdaptiveMaxPoolImpl<D, Derived>::AdaptiveMaxPoolImpl(
  const AdaptiveMaxPoolOptions<D>& options_) : options(options_) {}

template <size_t D, typename Derived>
void AdaptiveMaxPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void AdaptiveMaxPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AdaptiveMaxPool" << D << "d"
         << "(output_size=" << options.output_size() << ")";
}

Tensor AdaptiveMaxPool1dImpl::forward(const Tensor& input) {
  return F::adaptive_max_pool1d(input, options);
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool1dImpl::forward_with_indices(const Tensor& input) {
  return F::adaptive_max_pool1d_with_indices(input, options);
}

Tensor AdaptiveMaxPool2dImpl::forward(const Tensor& input) {
  return F::adaptive_max_pool2d(input, options);
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool2dImpl::forward_with_indices(const Tensor& input) {
  return F::adaptive_max_pool2d_with_indices(input, options);
}

Tensor AdaptiveMaxPool3dImpl::forward(const Tensor& input) {
  return F::adaptive_max_pool3d(input, options);
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool3dImpl::forward_with_indices(const Tensor& input) {
  return F::adaptive_max_pool3d_with_indices(input, options);
}

template class AdaptiveMaxPoolImpl<1, AdaptiveMaxPool1dImpl>;
template class AdaptiveMaxPoolImpl<2, AdaptiveMaxPool2dImpl>;
template class AdaptiveMaxPoolImpl<3, AdaptiveMaxPool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
AdaptiveAvgPoolImpl<D, Derived>::AdaptiveAvgPoolImpl(
  const AdaptiveAvgPoolOptions<D>& options_) : options(options_) {}

template <size_t D, typename Derived>
void AdaptiveAvgPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void AdaptiveAvgPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AdaptiveAvgPool" << D << "d"
         << "(output_size=" << options.output_size() << ")";
}

Tensor AdaptiveAvgPool1dImpl::forward(const Tensor& input) {
  return F::adaptive_avg_pool1d(input, options);
}

template class AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl>;

} // namespace nn
} // namespace torch
