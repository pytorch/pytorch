#include <torch/nn/modules/pooling.h>

#include <torch/expanding_array.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <tuple>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <size_t D>
MaxPoolOptions<D>::MaxPoolOptions(const ExpandingArray<D>& kernel_size)
    : kernel_size_(kernel_size), stride_(kernel_size) {}

template struct MaxPoolOptions<1>;
template struct MaxPoolOptions<2>;
template struct MaxPoolOptions<3>;

namespace detail {
template <size_t D, typename Derived>
MaxPoolImplBase<D, Derived>::MaxPoolImplBase(
    MaxPoolOptions<D> options,
    MaxPoolFunction max_pool)
    : FunctionalImpl([this, max_pool](Tensor input) {
        // ATen pooling functions return (output, indices).
        return std::get<0>(max_pool(
            input,
            this->options_.kernel_size_,
            this->options_.stride_,
            this->options_.padding_,
            this->options_.dilation_,
            this->options_.ceil_mode_));
      }),
      options_(options) {}

template <size_t D, typename Derived>
const MaxPoolOptions<D>& MaxPoolImplBase<D, Derived>::options() const noexcept {
  return options_;
}

template class MaxPoolImplBase<1, MaxPool1d>;
template class MaxPoolImplBase<2, MaxPool2d>;
template class MaxPoolImplBase<3, MaxPool3d>;

} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxPool1dImpl::MaxPool1dImpl(MaxPool1dOptions options)
    : detail::MaxPoolImplBase<1, MaxPool1dImpl>(options, at::max_pool1d) {}

MaxPool1dImpl::MaxPool1dImpl(const ExpandingArray<1>& kernel_size)
    : MaxPool1dImpl(MaxPool1dOptions(kernel_size)) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxPool2dImpl::MaxPool2dImpl(MaxPool2dOptions options)
    : detail::MaxPoolImplBase<2, MaxPool2dImpl>(options, at::max_pool2d) {}

MaxPool2dImpl::MaxPool2dImpl(const ExpandingArray<2>& kernel_size)
    : MaxPool2dImpl(MaxPool2dOptions(kernel_size)) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxPool3dImpl::MaxPool3dImpl(MaxPool3dOptions options)
    : detail::MaxPoolImplBase<3, MaxPool3dImpl>(options, at::max_pool3d) {}

MaxPool3dImpl::MaxPool3dImpl(const ExpandingArray<3>& kernel_size)
    : MaxPool3dImpl(MaxPool3dOptions(kernel_size)) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <size_t D>
AvgPoolOptions<D>::AvgPoolOptions(const ExpandingArray<D>& kernel_size)
    : kernel_size_(kernel_size), stride_(kernel_size) {}

template struct AvgPoolOptions<1>;
template struct AvgPoolOptions<2>;
template struct AvgPoolOptions<3>;

namespace detail {
template <size_t D, typename Derived>
AvgPoolImplBase<D, Derived>::AvgPoolImplBase(
    AvgPoolOptions<D> options,
    AvgPoolFunction avg_pool)
    : FunctionalImpl([this, avg_pool](Tensor input) {
        // ATen pooling functions return (output, indices).
        return avg_pool(
            input,
            this->options_.kernel_size_,
            this->options_.stride_,
            this->options_.padding_,
            this->options_.ceil_mode_,
            this->options_.count_include_pad_);
      }),
      options_(options) {}

template <size_t D, typename Derived>
const AvgPoolOptions<D>& AvgPoolImplBase<D, Derived>::options() const noexcept {
  return options_;
}

template class AvgPoolImplBase<1, AvgPool1d>;
template class AvgPoolImplBase<2, AvgPool2d>;
template class AvgPoolImplBase<3, AvgPool3d>;

} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AvgPool1dImpl::AvgPool1dImpl(AvgPool1dOptions options)
    : detail::AvgPoolImplBase<1, AvgPool1dImpl>(options, at::avg_pool1d) {}

AvgPool1dImpl::AvgPool1dImpl(const ExpandingArray<1>& kernel_size)
    : AvgPool1dImpl(AvgPool1dOptions(kernel_size)) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AvgPool2dImpl::AvgPool2dImpl(AvgPool2dOptions options)
    : detail::AvgPoolImplBase<2, AvgPool2dImpl>(options, at::avg_pool2d) {}

AvgPool2dImpl::AvgPool2dImpl(const ExpandingArray<2>& kernel_size)
    : AvgPool2dImpl(AvgPool2dOptions(kernel_size)) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AvgPool3dImpl::AvgPool3dImpl(AvgPool3dOptions options)
    : detail::AvgPoolImplBase<3, AvgPool3dImpl>(options, at::avg_pool3d) {}

AvgPool3dImpl::AvgPool3dImpl(const ExpandingArray<3>& kernel_size)
    : AvgPool3dImpl(AvgPool3dOptions(kernel_size)) {}
} // namespace nn
} // namespace torch
