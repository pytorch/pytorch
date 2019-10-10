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
struct TORCH_API L1LossImpl : Module {
  explicit L1LossImpl(const L1LossOptions& options_ = {});

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

/// Creates a criterion that measures the loss given an input tensor :math:`x`
/// and a labels tensor :math:`y` (containing 1 or -1). This is usually used for
/// measuring whether two inputs are similar or dissimilar, e.g. using the L1
/// pairwise distance as :math:`x`, and is typically used for learning nonlinear
/// embeddings or semi-supervised learning.
struct TORCH_API HingeEmbeddingLossImpl : Module {
  explicit HingeEmbeddingLossImpl(
      const HingeEmbeddingLossOptions& options_ = {});

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

// ============================================================================

/// Creates a criterion that optimizes a multi-class classification hinge
/// loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
/// output :math:`y` (which is a 1D tensor of target class indices,
/// :math:`0 \leq y \leq \text{x.size}(1)-1`):
struct TORCH_API MultiMarginLossImpl : Module {
  explicit MultiMarginLossImpl(
      const MultiMarginLossOptions& options_ = {});

  void reset();

  /// Pretty prints the `MultiMarginLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  MultiMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `MultiMarginLossImpl`.
/// See the documentation for `MultiMarginLossImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(MultiMarginLoss);

// ============================================================================

/// Creates a criterion that measures the loss given input tensors
/// `input1`, `input2`, and a `Tensor` label `target` with values 1 or
/// -1. This is used for measuring whether two inputs are similar or
/// dissimilar, using the cosine distance, and is typically used for learning
/// nonlinear embeddings or semi-supervised learning.
struct TORCH_API CosineEmbeddingLossImpl : Module {
  explicit CosineEmbeddingLossImpl(
      const CosineEmbeddingLossOptions& options_ = {});

  /// Pretty prints the `CosineEmbeddingLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(
      const Tensor& input1,
      const Tensor& input2,
      const Tensor& target);

  /// The options with which this `Module` was constructed.
  CosineEmbeddingLossOptions options;
};

/// A `ModuleHolder` subclass for `CosineEmbeddingLossImpl`.
/// See the documentation for `CosineEmbeddingLossImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(CosineEmbeddingLoss);

} // namespace nn
} // namespace torch
