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
struct TORCH_API L1LossImpl : public Cloneable<L1LossImpl> {
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

/// Creates a criterion that measures the loss given an input tensor :math:`x`
/// and a labels tensor :math:`y` (containing 1 or -1). This is usually used for
/// measuring whether two inputs are similar or dissimilar, e.g. using the L1
/// pairwise distance as :math:`x`, and is typically used for learning nonlinear
/// embeddings or semi-supervised learning.
struct TORCH_API HingeEmbeddingLossImpl : public Cloneable<HingeEmbeddingLossImpl> {
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

// ============================================================================

/// Creates a criterion that optimizes a multi-class classification hinge
/// loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
/// output :math:`y` (which is a 1D tensor of target class indices,
/// :math:`0 \leq y \leq \text{x.size}(1)-1`):
struct TORCH_API MultiMarginLossImpl : public Cloneable<MultiMarginLossImpl> {
  explicit MultiMarginLossImpl(
      const MultiMarginLossOptions& options_ = {});

  void reset() override;

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
struct TORCH_API CosineEmbeddingLossImpl : public Cloneable<CosineEmbeddingLossImpl> {
  explicit CosineEmbeddingLossImpl(
      const CosineEmbeddingLossOptions& options_ = {});

  void reset() override;

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

// ============================================================================

/// Creates a criterion that optimizes a multi-class multi-classification
/// hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
/// and output :math:`y` (which is a 2D `Tensor` of target class indices).
struct TORCH_API MultiLabelMarginLossImpl : public Cloneable<MultiLabelMarginLossImpl> {
  explicit MultiLabelMarginLossImpl(
    const MultiLabelMarginLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `L1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  MultiLabelMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `MultiLabelMarginLossImpl`.
/// See the documentation for `MultiLabelMarginLossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MultiLabelMarginLoss);

// ============================================================================

/// Creates a criterion that optimizes a two-class classification
/// logistic loss between input tensor :math:`x` and target tensor :math:`y`
/// (containing 1 or -1).
struct TORCH_API SoftMarginLossImpl : public Cloneable<SoftMarginLossImpl> {
  explicit SoftMarginLossImpl(const SoftMarginLossOptions& options_ = {});

  /// Pretty prints the `SoftMarginLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  SoftMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `SoftMarginLossImpl`.
/// See the documentation for `SoftMarginLossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(SoftMarginLoss);

// ============================================================================

/// Creates a criterion that optimizes a multi-label one-versus-all
/// loss based on max-entropy, between input :math:`x` and target :math:`y` of size
/// :math:`(N, C)`.
struct TORCH_API MultiLabelSoftMarginLossImpl : public Cloneable<MultiLabelSoftMarginLossImpl> {
  explicit MultiLabelSoftMarginLossImpl(
    const MultiLabelSoftMarginLossOptions& options_ = {});

  /// Pretty prints the `MultiLabelSoftMarginLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  MultiLabelSoftMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `MultiLabelSoftMarginLossImpl`.
/// See the documentation for `MultiLabelSoftMarginLossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MultiLabelSoftMarginLoss);

// ============================================================================

/// Creates a criterion that measures the triplet loss given an input
/// tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater 
/// than :math:`0`. This is used for measuring a relative similarity between
/// samples. A triplet is composed by `a`, `p` and `n` (i.e., `anchor`, 
/// `positive examples` and `negative examples` respectively). The
/// shapes of all input tensors should be :math:`(N, D)`
struct TORCH_API TripletMarginLossImpl : public Cloneable<TripletMarginLossImpl> {
  explicit TripletMarginLossImpl(
      const TripletMarginLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `TripletMarginLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(
      const Tensor& anchor,
      const Tensor& positive,
      const Tensor& negative);

  /// The options with which this `Module` was constructed.
  TripletMarginLossOptions options;
};

/// A `ModuleHolder` subclass for `TripletMarginLoss`.
/// See the documentation for `TripletMarginLossImpl` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(TripletMarginLoss);

} // namespace nn
} // namespace torch
