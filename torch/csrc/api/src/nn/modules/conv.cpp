#include <torch/nn/modules/conv.h>

#include <torch/expanding_array.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace torch { namespace nn {
template <size_t D, typename Derived>
Conv<D, Derived>::Conv(
    int64_t input_channels,
    int64_t output_channels,
    ExpandingArray<D> kernel_size)
    : input_channels_(input_channels),
      output_channels_(output_channels),
      kernel_size_(std::move(kernel_size)) {}

template <size_t D, typename Derived>
void Conv<D, Derived>::reset() {
  if (!transposed_) {
    for (auto pad : *output_padding_) {
      AT_CHECK(
          pad == 0, "Only transposed convolutions support output padding!");
    }
  }

  std::vector<int64_t> weights_size;
  if (transposed_) {
    weights_size.push_back(input_channels_);
    weights_size.push_back(output_channels_ / groups_);
  } else {
    weights_size.push_back(output_channels_);
    weights_size.push_back(input_channels_ / groups_);
  }
  weights_size.insert(
      weights_size.end(), kernel_size_->begin(), kernel_size_->end());
  AT_ASSERT(weights_size.size() == 2 + kernel_size_->size());

  weight_ = this->add(Var(at::CPU(at::kFloat).empty(weights_size)), "weight");
  if (with_bias_) {
    bias_ = this->add(Var(at::CPU(at::kFloat).empty(output_channels_)), "bias");
  }

  const auto number_of_features = std::accumulate(
      kernel_size_->begin(),
      kernel_size_->end(),
      input_channels_,
      std::multiplies<int64_t>{});
  const auto stdv = 1.0 / std::sqrt(number_of_features);
  for (auto& p : this->parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Conv1d::forward(variable_list input) {
  AT_ASSERT(input.front().ndimension() == 3);

  if (transposed_) {
    return {at::conv_transpose1d(
        input.front(),
        weight_,
        bias_,
        stride_,
        padding_,
        output_padding_,
        groups_,
        dilation_)};
  }
  return {at::conv1d(
      input.front(), weight_, bias_, stride_, padding_, dilation_, groups_)};
}

variable_list Conv2d::forward(variable_list input) {
  AT_ASSERT(input.front().ndimension() == 4);

  if (transposed_) {
    return {at::conv_transpose2d(
        input.front(),
        weight_,
        bias_,
        stride_,
        padding_,
        output_padding_,
        groups_,
        dilation_)};
  }
  return {at::conv2d(
      input.front(), weight_, bias_, stride_, padding_, dilation_, groups_)};
}

variable_list Conv3d::forward(variable_list input) {
  AT_ASSERT(input.front().ndimension() == 5);

  if (transposed_) {
    return {at::conv_transpose3d(
        input.front(),
        weight_,
        bias_,
        stride_,
        padding_,
        output_padding_,
        groups_,
        dilation_)};
  } else {
    return {at::conv3d(
        input.front(), weight_, bias_, stride_, padding_, dilation_, groups_)};
  }
}

template class Conv<1, Conv1d>;
template class Conv<2, Conv2d>;
template class Conv<3, Conv3d>;

}} // namespace torch::nn
