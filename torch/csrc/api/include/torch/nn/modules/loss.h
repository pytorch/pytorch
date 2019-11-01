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

/// Creates a criterion that uses a squared term if the absolute
/// element-wise error falls below 1 and an L1 term otherwise.
/// It is less sensitive to outliers than the `MSELoss` and in some cases
/// prevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).
/// Also known as the Huber loss.
struct TORCH_API SmoothL1LossImpl : public Cloneable<SmoothL1LossImpl> {
  explicit SmoothL1LossImpl(const SmoothL1LossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `L1Loss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  SmoothL1LossOptions options;
};

/// A `ModuleHolder` subclass for `SmoothL1LossImpl`.
/// See the documentation for `SmoothL1LossImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(SmoothL1Loss);

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

// ============================================================================

/// Calculates loss between a continuous (unsegmented) time series and a target
/// sequence. CTCLoss sums over the probability of possible alignments of input
/// to target, producing a loss value which is differentiable with respect
/// to each input node. The alignment of input to target is assumed
/// to be "many-to-one", which limits the length of the target sequence
/// such that it must be less than or equal to the input length.
struct TORCH_API CTCLossImpl : public Cloneable<CTCLossImpl> {

  explicit CTCLossImpl(const CTCLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `CTCLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& log_probs, const Tensor& targets,
                 const Tensor& input_lengths, const Tensor& target_lengths);

  /// The options with which this `Module` was constructed.
  CTCLossOptions options;
};

/// A `ModuleHolder` subclass for `CTCLossImpl`.
/// See the documentation for `CTCLoss` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(CTCLoss);

// ============================================================================

struct TORCH_API PoissonNLLLossImpl : public Cloneable<PoissonNLLLossImpl> {
  explicit PoissonNLLLossImpl(const PoissonNLLLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `PoissonNLLLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& log_input, const Tensor& targets);

  /// The options with which this `Module` was constructed.
  PoissonNLLLossOptions options;
};

/// A `ModuleHolder` subclass for `PoissonNLLLossImpl`.
/// See the documentation for `PoissonNLLLoss` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(PoissonNLLLoss);

// ============================================================================

struct TORCH_API MarginRankingLossImpl : public Cloneable<MarginRankingLossImpl> {
  explicit MarginRankingLossImpl(const MarginRankingLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `MarginRankingLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1,
    const Tensor& input2, const Tensor& targets);

  /// The options with which this `Module` was constructed.
  MarginRankingLossOptions options;
};

/// A `ModuleHolder` subclass for `MarginRankingLossImpl`.
/// See the documentation for `MarginRankingLoss` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(MarginRankingLoss);

// ============================================================================

struct TORCH_API BCEWithLogitsLossImpl : public Cloneable<BCEWithLogitsLossImpl> {
  explicit BCEWithLogitsLossImpl(const BCEWithLogitsLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `BCEWithLogitsLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  BCEWithLogitsLossOptions options;
};

/// A `ModuleHolder` subclass for `BCEWithLogitsLossImpl`.
/// See the documentation for `BCEWithLogitsLoss` class to learn what
/// methods it provides, or the documentation for `ModuleHolder` to learn about
/// PyTorch's module storage semantics.
TORCH_MODULE(BCEWithLogitsLoss);

} // namespace nn
} // namespace torch
