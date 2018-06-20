#include <torch/nn/modules/conv.h>

#include <torch/expanding_array.h>
#include <torch/functions.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

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

std::vector<Variable> Conv1dImpl::forward(std::vector<Variable> input) {
  AT_ASSERT(input.front().ndimension() == 3);

  if (options_.transposed_) {
    return {at::conv_transpose1d(
        input.front(),
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_)};
  }
  return {at::conv1d(
      input.front(),
      weight_,
      bias_,
      options_.stride_,
      options_.padding_,
      options_.dilation_,
      options_.groups_)};
}

std::vector<Variable> Conv2dImpl::forward(std::vector<Variable> input) {
  AT_ASSERT(input.front().ndimension() == 4);

  if (options_.transposed_) {
    return {at::conv_transpose2d(
        input.front(),
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_)};
  }
  return {at::conv2d(
      input.front(),
      weight_,
      bias_,
      options_.stride_,
      options_.padding_,
      options_.dilation_,
      options_.groups_)};
}

std::vector<Variable> Conv3dImpl::forward(std::vector<Variable> input) {
  AT_ASSERT(input.front().ndimension() == 5);

  if (options_.transposed_) {
    return {at::conv_transpose3d(
        input.front(),
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.output_padding_,
        options_.groups_,
        options_.dilation_)};
  } else {
    return {at::conv3d(
        input.front(),
        weight_,
        bias_,
        options_.stride_,
        options_.padding_,
        options_.dilation_,
        options_.groups_)};
  }
}

#define CONV_D(D)                 \
  template struct ConvOptions<D>; \
  template class ConvImpl<D, Conv##D##dImpl>

CONV_D(1);
CONV_D(2);
CONV_D(3);

#undef CONV_D

} // namespace nn
} // namespace torch
