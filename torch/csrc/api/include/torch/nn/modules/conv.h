#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
template <size_t D>
struct ConvOptions {
  ConvOptions(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<D> kernel_size);

  TORCH_ARG(int64_t, input_channels);
  TORCH_ARG(int64_t, output_channels);
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  TORCH_ARG(ExpandingArray<D>, stride) = 1;
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;
  TORCH_ARG(bool, transposed) = false;
  TORCH_ARG(bool, with_bias) = true;
  TORCH_ARG(int64_t, groups) = 1;
};

template <size_t D, typename Derived>
class ConvImpl : public torch::nn::Cloneable<Derived> {
 public:
  explicit ConvImpl(ConvOptions<D> options);

  void reset() override;

  ConvOptions<D> options;
  Tensor weight;
  Tensor bias;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Conv1dImpl : public ConvImpl<1, Conv1dImpl> {
 public:
  using ConvImpl<1, Conv1dImpl>::ConvImpl;
  Tensor forward(Tensor input);
};
using Conv1dOptions = ConvOptions<1>;
TORCH_MODULE(Conv1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Conv2dImpl : public ConvImpl<2, Conv2dImpl> {
 public:
  using ConvImpl<2, Conv2dImpl>::ConvImpl;
  Tensor forward(Tensor input);
};
using Conv2dOptions = ConvOptions<2>;
TORCH_MODULE(Conv2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Conv3dImpl : public ConvImpl<3, Conv3dImpl> {
 public:
  using ConvImpl<3, Conv3dImpl>::ConvImpl;
  Tensor forward(Tensor input);
};
using Conv3dOptions = ConvOptions<3>;
TORCH_MODULE(Conv3d);

} // namespace nn
} // namespace torch
