#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/options/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Creates a criterion that measures the mean absolute error (MAE) between each
/// element in the input : math :`x` and target : `y`.
struct TORCH_API L1LossImpl : Cloneable<L1LossImpl> {
  explicit L1LossImpl(const L1LossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `L1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  L1LossOptions options;
};

/// A `ModuleHolder` subclass for `L1LossImpl`.
/// See the documentation for `L1LossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(L1Loss);

// ============================================================================
/// The `Kullback-Leibler divergence`_ Loss
///
/// KL divergence is a useful distance measure for continuous distributions
/// and is often useful when performing direct regression over the space of
/// (discretely sampled) continuous output distributions.
///
/// As with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain
/// *log-probabilities* and is not restricted to a 2D Tensor.
/// The targets are given as *probabilities* (i.e. without taking the
/// logarithm).
///
/// This criterion expects a `target` `Tensor` of the same size as the
/// `input` `Tensor`.
struct TORCH_API KLDivLossImpl : Cloneable<KLDivLossImpl> {
  explicit KLDivLossImpl(const KLDivLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `KLDivLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  KLDivLossOptions options;
};

/// A `ModuleHolder` subclass for `KLDivLossImpl`.
/// See the documentation for `KLDivLossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(KLDivLoss);

// ============================================================================

/// Creates a criterion that measures the mean squared error (squared L2 norm)
/// between each element in the input :math:`x` and target :math:`y`.
struct TORCH_API MSELossImpl : Cloneable<MSELossImpl> {
  explicit MSELossImpl(const MSELossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `MSELoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  MSELossOptions options;
};

/// A `ModuleHolder` subclass for `MSELossImpl`.
/// See the documentation for `MSELossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MSELoss);

// ============================================================================

/// Creates a criterion that measures the Binary Cross Entropy
/// between the target and the output.
struct TORCH_API BCELossImpl : Cloneable<BCELossImpl> {
  explicit BCELossImpl(const BCELossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `BCELoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  BCELossOptions options;
};

/// A `ModuleHolder` subclass for `BCELossImpl`.
/// See the documentation for `BCELossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BCELoss);

// ============================================================================

/// Creates a criterion that measures the loss given an input tensor :math:`x`
/// and a labels tensor :math:`y` (containing 1 or -1). This is usually used for
/// measuring whether two inputs are similar or dissimilar, e.g. using the L1
/// pairwise distance as :math:`x`, and is typically used for learning nonlinear
/// embeddings or semi-supervised learning.
struct TORCH_API HingeEmbeddingLossImpl : Cloneable<HingeEmbeddingLossImpl> {
  explicit HingeEmbeddingLossImpl(
      const HingeEmbeddingLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `HingeEmbeddingLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  HingeEmbeddingLossOptions options;
};

/// A `ModuleHolder` subclass for `HingeEmbeddingLossImpl`.
/// See the documentation for `HingeEmbeddingLossImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(HingeEmbeddingLoss);

} // namespace nn
} // namespace torch
