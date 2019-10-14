#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/layernorm.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>

namespace torch {
namespace nn {

class TORCH_API LayerNormImpl : public torch::nn::Cloneable<LayerNormImpl> {
 public:
  explicit LayerNormImpl(torch::IntArrayRef normalized_shape)
      : LayerNormImpl(LayerNormOptions(normalized_shape)) {}
  explicit LayerNormImpl(const LayerNormOptions& options_);

  void reset() override;

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies layer normalization over a mini-batch of inputs as described in
  /// the paper `Layer Normalization`_ .
  ///
  /// The mean and standard-deviation are calculated separately over the last
  /// certain number dimensions which have to be of the shape specified by
  /// input `normalized_shape`.
  ///
  /// `Layer Normalization`: https://arxiv.org/abs/1607.06450
  Tensor forward(const Tensor& input);

  /// The options with which this module was constructed.
  LayerNormOptions options;

  /// The learned weight.
  /// Initialized to ones if the `elementwise_affine` option is set to `true` upon construction.
  Tensor weight;

  /// The learned bias.
  /// Initialized to zeros `elementwise_affine` option is set to `true` upon construction.
  Tensor bias;
};

  //todo- update description
  /// A `ModuleHolder` subclass for `LayerNormImpl`.
  /// See the documentation for `LayerNormImpl` class to learn what methods it
  /// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
  /// module storage semantics.
  TORCH_MODULE(LayerNorm);
};

} // namespace nn
} // namespace torch
