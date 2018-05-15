#pragma once

#include <torch/nn/module.h>

#include <cstdint>

namespace torch { namespace nn {
class Conv : public torch::nn::CloneableModule<Conv> {
 public:
  // TODO: Create a type that can be implicitly constructed from a vector, or an
  // int, and does the right thing. Then we can remove one overload here.
  Conv(
      uint32_t Nd,
      uint32_t in_chan,
      uint32_t out_chan,
      IntVec ks,
      bool transposed = false,
      bool with_bias = true,
      int groups = 1);

  Conv(
      uint32_t Nd,
      uint32_t in_chan,
      uint32_t out_chan,
      int ks,
      bool transposed = false,
      bool with_bias = true,
      int groups = 1);

  variable_list forward(variable_list) override;

  Conv& stride(size_t value);
  Conv& padding(size_t value);
  Conv& dilation(size_t value);
  Conv& output_padding(size_t value);

  Variable weight, bias;
  uint32_t Nd_;
  uint32_t in_channels_;
  uint32_t out_channels_;
  bool transposed_;
  int groups_;
  IntVec ks_;
  IntVec stride_;
  IntVec padding_;
  IntVec dilation_;
  bool dilated_;
  IntVec output_padding_;
};

#define CONV_D(D)                                                          \
  class Conv##D##d : public Conv {                                         \
   public:                                                                 \
    Conv##D##d(uint32_t i, uint32_t o, int ks) : Conv((D), i, o, ks) {}    \
    Conv##D##d(uint32_t i, uint32_t o, IntVec ks) : Conv((D), i, o, ks) {} \
  }

CONV_D(1);
CONV_D(2);
CONV_D(3);

#undef CONV_D
}} // namespace torch::nn
