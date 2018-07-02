#include <torch/nn/modules/conv.h>

#include <torch/expanding_array.h>
#include <torch/tensor.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace torch {
namespace nn {
template <size_t D>
ConvOptions<D>::ConvOptions(
    int64_t input_channels,
    int64_t output_channels,
    ExpandingArray<D> kernel_size)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(std::move(kernel_size)) {}

template <size_t D, typename Derived>
ConvImpl<D, Derived>::ConvImpl(ConvOptions<D> options)
    : options_(std::move(options)) {
  reset();
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::reset() {
  if (!options_.transposed_) {
    for (auto pad : *options_.output_padding_) {
      AT_CHECK(
          pad == 0, "Only transposed convolutions support output padding!");
    }
  }

  std::vector<int64_t> weights_size;
  if (options_.transposed_) {
    weights_size.push_back(options_.input_channels_);
    weights_size.push_back(options_.output_channels_ / options_.groups_);
  } else {
    weights_size.push_back(options_.output_channels_);
    weights_size.push_back(options_.input_channels_ / options_.groups_);
  }
  weights_size.insert(
      weights_size.end(),
      options_.kernel_size_->begin(),
      options_.kernel_size_->end());
  AT_ASSERT(weights_size.size() == 2 + options_.kernel_size_->size());

  weight_ = this->register_parameter("weight", torch::empty(weights_size));
  if (options_.with_bias_) {
    bias_ = this->register_parameter(
        "bias", torch::empty(options_.output_channels_));
  }

  const auto number_of_features = std::accumulate(
      options_.kernel_size_->begin(),
      options_.kernel_size_->end(),
      options_.input_channels_,
      std::multiplies<int64_t>{});
  const auto stdv = 1.0 / std::sqrt(number_of_features);
  for (auto& p : this->parameters()) {
    p->data().uniform_(-stdv, stdv);
  }
}

template <size_t D, typename Derived>
const ConvOptions<D>& ConvImpl<D, Derived>::options() const noexcept {
  return options_;
}

Tensor Conv1dImpl::forward(Tensor input) {
  AT_ASSERT(input.ndimension() == 3);

  if (options_.transposed_) {
    return torch::conv_transpose1d(
        input,
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_);
  }
  return torch::conv1d(
      input,
      weight_,
      bias_,
      options_.stride_,
      options_.padding_,
      options_.dilation_,
      options_.groups_);
}

Tensor Conv2dImpl::forward(Tensor input) {
  AT_ASSERT(input.ndimension() == 4);

  if (options_.transposed_) {
    return torch::conv_transpose2d(
        input,
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_);
  }
  return torch::conv2d(
      input,
      weight_,
      bias_,
      options_.stride_,
      options_.padding_,
      options_.dilation_,
      options_.groups_);
}

Tensor Conv3dImpl::forward(Tensor input) {
  AT_ASSERT(input.ndimension() == 5);

  if (options_.transposed_) {
    return torch::conv_transpose3d(
        input,
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_);
  } else {
    return torch::conv3d(
        input,
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.dilation_,
        options_.groups_);
  }
}

template struct ConvOptions<1>;
template class ConvImpl<1, Conv1dImpl>;

template struct ConvOptions<2>;
template class ConvImpl<2, Conv2dImpl>;

template struct ConvOptions<3>;
template class ConvImpl<3, Conv3dImpl>;

} // namespace nn
} // namespace torch
