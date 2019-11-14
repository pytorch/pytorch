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
         << ", stride=" << options.stride()
         << ", padding=" << options.padding() << ")";
}

Tensor AvgPool1dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool1d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad());
}

Tensor AvgPool2dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool2d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad(),
    options.divisor_override());
}

Tensor AvgPool3dImpl::forward(const Tensor& input) {
  return F::detail::avg_pool3d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad(),
    options.divisor_override());
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
  stream << std::boolalpha
         << "torch::nn::MaxPool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding()
         << ", dilation=" << options.dilation()
         << ", ceil_mode=" << options.ceil_mode() << ")";
}

Tensor MaxPool1dImpl::forward(const Tensor& input) {
  return F::detail::max_pool1d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
}

std::tuple<Tensor, Tensor> MaxPool1dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::max_pool1d_with_indices(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
}

Tensor MaxPool2dImpl::forward(const Tensor& input) {
  return F::detail::max_pool2d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
}

std::tuple<Tensor, Tensor> MaxPool2dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::max_pool2d_with_indices(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
}

Tensor MaxPool3dImpl::forward(const Tensor& input) {
  return F::detail::max_pool3d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
}

std::tuple<Tensor, Tensor> MaxPool3dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::max_pool3d_with_indices(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.ceil_mode());
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
  return F::detail::adaptive_max_pool1d(input, options.output_size());
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool1dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::adaptive_max_pool1d_with_indices(input, options.output_size());
}

Tensor AdaptiveMaxPool2dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_max_pool2d(input, options.output_size());
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool2dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::adaptive_max_pool2d_with_indices(input, options.output_size());
}

Tensor AdaptiveMaxPool3dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_max_pool3d(input, options.output_size());
}

std::tuple<Tensor, Tensor> AdaptiveMaxPool3dImpl::forward_with_indices(const Tensor& input) {
  return F::detail::adaptive_max_pool3d_with_indices(input, options.output_size());
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
  return F::detail::adaptive_avg_pool1d(input, options.output_size());
}

Tensor AdaptiveAvgPool2dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_avg_pool2d(input, options.output_size());
}

Tensor AdaptiveAvgPool3dImpl::forward(const Tensor& input) {
  return F::detail::adaptive_avg_pool3d(input, options.output_size());
}

template class AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl>;
template class AdaptiveAvgPoolImpl<2, AdaptiveAvgPool2dImpl>;
template class AdaptiveAvgPoolImpl<3, AdaptiveAvgPool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
MaxUnpoolImpl<D, Derived>::MaxUnpoolImpl(const MaxUnpoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void MaxUnpoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void MaxUnpoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::MaxUnpool" << D << "d"
         << "(kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding() << ")";
}

Tensor MaxUnpool1dImpl::forward(const Tensor& input, const Tensor& indices,
    const c10::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool1d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    output_size);
}

Tensor MaxUnpool2dImpl::forward(const Tensor& input, const Tensor& indices,
    const c10::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool2d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    output_size);
}

Tensor MaxUnpool3dImpl::forward(const Tensor& input, const Tensor& indices,
    const c10::optional<std::vector<int64_t>>& output_size) {
  return F::detail::max_unpool3d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    output_size);
}

template class MaxUnpoolImpl<1, MaxUnpool1dImpl>;
template class MaxUnpoolImpl<2, MaxUnpool2dImpl>;
template class MaxUnpoolImpl<3, MaxUnpool3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
LPPoolImpl<D, Derived>::LPPoolImpl(const LPPoolOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void LPPoolImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
void LPPoolImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::LPPool" << D << "d("
         << "norm_type=" << options.norm_type() << ", "
         << "kernel_size=" << options.kernel_size() << ", "
         << "stride=" << options.stride() << ", "
         << "ceil_mode=" << options.ceil_mode() << ")";
}

Tensor LPPool1dImpl::forward(const Tensor& input) {
  return F::detail::lp_pool1d(
    input,
    options.norm_type(),
    options.kernel_size(),
    options.stride(),
    options.ceil_mode());
}

template class LPPoolImpl<1, LPPool1dImpl>;

Tensor LPPool2dImpl::forward(const Tensor& input) {
  return F::detail::lp_pool2d(
    input,
    options.norm_type(),
    options.kernel_size(),
    options.stride(),
    options.ceil_mode());
}

template class LPPoolImpl<2, LPPool2dImpl>;

} // namespace nn
} // namespace torch
