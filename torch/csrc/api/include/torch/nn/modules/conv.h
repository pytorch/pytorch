#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/module.h>

#include <cstdint>

namespace torch { namespace nn {

template <size_t D, typename Derived>
class Conv : public torch::nn::CloneableModule<Derived> {
 public:
  Conv(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size);

  void reset() override;

  TORCH_ATTR(int64_t, input_channels);
  TORCH_ATTR(int64_t, output_channels);
  TORCH_ATTR(ExpandingArray<D>, kernel_size);
  TORCH_ATTR(ExpandingArray<D>, stride) = 1;
  TORCH_ATTR(ExpandingArray<D>, padding) = 0;
  TORCH_ATTR(ExpandingArray<D>, dilation) = 1;
  TORCH_ATTR(ExpandingArray<D>, output_padding) = 0;
  TORCH_ATTR(bool, transposed) = false;
  TORCH_ATTR(bool, with_bias) = true;
  TORCH_ATTR(int64_t, groups) = 1;
  TORCH_ATTR(Variable, weight);
  TORCH_ATTR(Variable, bias);
};

#define CONV_D(dimensions)                                                     \
  class Conv##dimensions##d : public Conv<(dimensions), Conv##dimensions##d> { \
   public:                                                                     \
    using Conv<(dimensions), Conv##dimensions##d>::Conv;                       \
    variable_list forward(variable_list) override;                             \
  }

CONV_D(1);
CONV_D(2);
CONV_D(3);

#undef CONV_D

}} // namespace torch::nn
