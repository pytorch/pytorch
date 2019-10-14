#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `LayerNorm` module.
struct TORCH_API LayerNormOptions {
  /* implicit */ LayerNormOptions(torch::IntArrayRef normalized_shape);
  /// input shape from an expected input.
  TORCH_ARG(torch::IntArrayRef, normalized_shape);
  /// a boolean value that when set to ``True``, this module
  /// has learnable per-element affine parameters initialized to ones (for weights)
  /// and zeros (for biases).
  TORCH_ARG(bool, elementwise_affine) = true;
  /// a value added to the denominator for numerical stability.
  TORCH_ARG(double, eps) = 1e-5;

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
