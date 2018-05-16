#pragma once

#include <torch/nn/module.h>

#include <ATen/ScalarType.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace torch { namespace nn {

template <size_t D, typename Derived>
class Conv : public torch::nn::CloneableModule<Derived> {
 public:
  struct ExpandingSize {
    ExpandingSize(std::initializer_list<int64_t> list);
    ExpandingSize(std::vector<int64_t> sizes);
    ExpandingSize(int64_t single_size);
    std::array<int64_t, D>& operator*();
    std::array<int64_t, D>* operator->();
    operator at::IntList();
    std::array<int64_t, D> sizes_;
  };

  Conv(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingSize kernel_size);

  void reset() override;

  TORCH_PARAMETER(int64_t, input_channels);
  TORCH_PARAMETER(int64_t, output_channels);
  TORCH_PARAMETER(ExpandingSize, kernel_size);
  TORCH_PARAMETER(ExpandingSize, stride) = 1;
  TORCH_PARAMETER(ExpandingSize, padding) = 0;
  TORCH_PARAMETER(ExpandingSize, dilation) = 1;
  TORCH_PARAMETER(ExpandingSize, output_padding) = 0;
  TORCH_PARAMETER(bool, transposed) = false;
  TORCH_PARAMETER(bool, with_bias) = true;
  TORCH_PARAMETER(int64_t, groups) = 1;

 protected:
  Variable weight_;
  Variable bias_;
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
