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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ L1Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the mean absolute error (MAE) between each
/// element in the input : math :`x` and target : `y`.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.L1Loss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::L1LossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// L1Loss model(L1LossOptions(torch::kNone));
/// ```
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
/// provides, and examples of how to use `L1Loss` with `torch::nn::L1LossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(L1Loss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ KLDivLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The Kullback-Leibler divergence loss measure
/// See https://pytorch.org/docs/master/nn.html#torch.nn.KLDivLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::KLDivLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// KLDivLoss model(KLDivLossOptions().reduction(torch::kNone));
/// ```
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
/// provides, and examples of how to use `KLDivLoss` with `torch::nn::KLDivLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(KLDivLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MSELoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the mean squared error (squared L2 norm)
/// between each element in the input :math:`x` and target :math:`y`.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MSELoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MSELossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MSELoss model(MSELossOptions(torch::kNone));
/// ```
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
/// provides, and examples of how to use `MSELoss` with `torch::nn::MSELossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MSELoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCELoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the Binary Cross Entropy
/// between the target and the output.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BCELoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BCELossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BCELoss model(BCELossOptions().reduction(torch::kNone).weight(weight));
/// ```
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
/// provides, and examples of how to use `BCELoss` with `torch::nn::BCELossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BCELoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HingeEmbeddingLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the loss given an input tensor :math:`x`
/// and a labels tensor :math:`y` (containing 1 or -1).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.HingeEmbeddingLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::HingeEmbeddingLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// HingeEmbeddingLoss model(HingeEmbeddingLossOptions().margin(4).reduction(torch::kNone));
/// ```
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
/// See the documentation for `HingeEmbeddingLossImpl` class to learn what methods it
/// provides, and examples of how to use `HingeEmbeddingLoss` with `torch::nn::HingeEmbeddingLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(HingeEmbeddingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiMarginLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that optimizes a multi-class classification hinge
/// loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
/// output :math:`y` (which is a 1D tensor of target class indices,
/// :math:`0 \leq y \leq \text{x.size}(1)-1`).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MultiMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MultiMarginLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiMarginLoss model(MultiMarginLossOptions().margin(2).weight(weight));
/// ```
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
/// See the documentation for `MultiMarginLossImpl` class to learn what methods it
/// provides, and examples of how to use `MultiMarginLoss` with `torch::nn::MultiMarginLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MultiMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CosineEmbeddingLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the loss given input tensors
/// `input1`, `input2`, and a `Tensor` label `target` with values 1 or
/// -1. This is used for measuring whether two inputs are similar or
/// dissimilar, using the cosine distance, and is typically used for learning
/// nonlinear embeddings or semi-supervised learning.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.CosineEmbeddingLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::CosineEmbeddingLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CosineEmbeddingLoss model(CosineEmbeddingLossOptions().margin(0.5));
/// ```
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
/// See the documentation for `CosineEmbeddingLossImpl` class to learn what methods it
/// provides, and examples of how to use `CosineEmbeddingLoss` with `torch::nn::CosineEmbeddingLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(CosineEmbeddingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SmoothL1Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that uses a squared term if the absolute
/// element-wise error falls below beta and an L1 term otherwise.
/// It is less sensitive to outliers than the `MSELoss` and in some cases
/// prevents exploding gradients (e.g. see the paper `Fast R-CNN` by Ross Girshick).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::SmoothL1LossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// SmoothL1Loss model(SmoothL1LossOptions().reduction(torch::kNone).beta(0.5));
/// ```
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
/// provides, and examples of how to use `SmoothL1Loss` with `torch::nn::SmoothL1LossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(SmoothL1Loss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HuberLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that uses a squared term if the absolute
/// element-wise error falls below delta and a delta-scaled L1 term otherwise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.HuberLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::HuberLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// HuberLoss model(HuberLossOptions().reduction(torch::kNone).delta(0.5));
/// ```
struct TORCH_API HuberLossImpl : public Cloneable<HuberLossImpl> {
  explicit HuberLossImpl(const HuberLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `HuberLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  HuberLossOptions options;
};

/// A `ModuleHolder` subclass for `HuberLossImpl`.
/// See the documentation for `HuberLossImpl` class to learn what methods it
/// provides, and examples of how to use `HuberLoss` with `torch::nn::HuberLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(HuberLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiLabelMarginLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that optimizes a multi-class multi-classification
/// hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
/// and output :math:`y` (which is a 2D `Tensor` of target class indices).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MultiLabelMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MultiLabelMarginLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiLabelMarginLoss model(MultiLabelMarginLossOptions(torch::kNone));
/// ```
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
/// provides, and examples of how to use `MultiLabelMarginLoss` with `torch::nn::MultiLabelMarginLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MultiLabelMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftMarginLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that optimizes a two-class classification
/// logistic loss between input tensor :math:`x` and target tensor :math:`y`
/// (containing 1 or -1).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.SoftMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::SoftMarginLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// SoftMarginLoss model(SoftMarginLossOptions(torch::kNone));
/// ```
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
/// provides, and examples of how to use `SoftMarginLoss` with `torch::nn::SoftMarginLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(SoftMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiLabelSoftMarginLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that optimizes a multi-label one-versus-all
/// loss based on max-entropy, between input :math:`x` and target :math:`y` of size
/// :math:`(N, C)`.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MultiLabelSoftMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MultiLabelSoftMarginLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MultiLabelSoftMarginLoss model(MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight));
/// ```
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
/// provides, and examples of how to use `MultiLabelSoftMarginLoss` with `torch::nn::MultiLabelSoftMarginLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MultiLabelSoftMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TripletMarginLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the triplet loss given an input
/// tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater
/// than :math:`0`. This is used for measuring a relative similarity between
/// samples. A triplet is composed by `a`, `p` and `n` (i.e., `anchor`,
/// `positive examples` and `negative examples` respectively). The
/// shapes of all input tensors should be :math:`(N, D)`.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.TripletMarginLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::TripletMarginLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// TripletMarginLoss model(TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false));
/// ```
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

/// A `ModuleHolder` subclass for `TripletMarginLossImpl`.
/// See the documentation for `TripletMarginLossImpl` class to learn what methods it
/// provides, and examples of how to use `TripletMarginLoss` with `torch::nn::TripletMarginLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TripletMarginLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TripletMarginWithDistanceLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the triplet loss given input
/// tensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,
/// positive, and negative examples, respectively); and a nonnegative, real-valued function
/// ("distance function") used to compute the relationships between the anchor
/// and positive example ("positive distance") and the anchor and negative
/// example ("negative distance").
/// See https://pytorch.org/docs/master/nn.html#torch.nn.TripletMarginWithDistanceLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::TripletMarginWithDistanceLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// TripletMarginWithDistanceLoss model(TripletMarginWithDistanceLossOptions().margin(3).swap(false));
/// ```
struct TORCH_API TripletMarginWithDistanceLossImpl : public Cloneable<TripletMarginWithDistanceLossImpl> {
  explicit TripletMarginWithDistanceLossImpl(
      TripletMarginWithDistanceLossOptions options_ = {});

  void reset() override;

  /// Pretty prints the `TripletMarginWithDistanceLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(
      const Tensor& anchor,
      const Tensor& positive,
      const Tensor& negative);

  /// The options with which this `Module` was constructed.
  TripletMarginWithDistanceLossOptions options;
};

/// A `ModuleHolder` subclass for `TripletMarginWithDistanceLossImpl`.
/// See the documentation for `TripletMarginWithDistanceLossImpl` class to learn what methods it
/// provides, and examples of how to use `TripletMarginWithDistanceLoss` with
/// `torch::nn::TripletMarginWithDistanceLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TripletMarginWithDistanceLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CTCLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The Connectionist Temporal Classification loss.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.CTCLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::CTCLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CTCLoss model(CTCLossOptions().blank(42).zero_infinity(false).reduction(torch::kSum));
/// ```
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
/// See the documentation for `CTCLossImpl` class to learn what methods it
/// provides, and examples of how to use `CTCLoss` with `torch::nn::CTCLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(CTCLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PoissonNLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Negative log likelihood loss with Poisson distribution of target.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.PoissonNLLLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::PoissonNLLLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// PoissonNLLLoss model(PoissonNLLLossOptions().log_input(false).full(true).eps(0.42).reduction(torch::kSum));
/// ```
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
/// See the documentation for `PoissonNLLLossImpl` class to learn what methods it
/// provides, and examples of how to use `PoissonNLLLoss` with `torch::nn::PoissonNLLLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(PoissonNLLLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MarginRankingLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that measures the loss given
/// inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
/// and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MarginRankingLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::MarginRankingLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// MarginRankingLoss model(MarginRankingLossOptions().margin(0.5).reduction(torch::kSum));
/// ```
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
/// See the documentation for `MarginRankingLossImpl` class to learn what methods it
/// provides, and examples of how to use `MarginRankingLoss` with `torch::nn::MarginRankingLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MarginRankingLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The negative log likelihood loss. It is useful to train a classification
/// problem with `C` classes.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::NLLLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// NLLLoss model(NLLLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
struct TORCH_API NLLLossImpl : public Cloneable<NLLLossImpl> {
  explicit NLLLossImpl(
      const NLLLossOptions& options_ = {});

  /// Pretty prints the `NLLLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  void reset() override;

  Tensor forward(
      const Tensor& input,
      const Tensor& target);

  /// The options with which this `Module` was constructed.
  NLLLossOptions options;

  /// A manual rescaling weight given to to each class.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `NLLLossImpl`.
/// See the documentation for `NLLLossImpl` class to learn what methods it
/// provides, and examples of how to use `NLLLoss` with `torch::nn::NLLLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(NLLLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CrossEntropyLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates a criterion that combines :func:`nn.LogSoftmax` and
/// :func:`nn.NLLLoss` in one single class.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::CrossEntropyLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CrossEntropyLoss model(CrossEntropyLossOptions().ignore_index(-100).reduction(torch::kMean));
/// ```
struct TORCH_API CrossEntropyLossImpl : public Cloneable<CrossEntropyLossImpl> {
  explicit CrossEntropyLossImpl(
      const CrossEntropyLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `CrossEntropyLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(
      const Tensor& input,
      const Tensor& target);

  /// The options with which this `Module` was constructed.
  CrossEntropyLossOptions options;

  /// A manual rescaling weight given to to each class.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `CrossEntropyLossImpl`.
/// See the documentation for `CrossEntropyLossImpl` class to learn what methods it
/// provides, and examples of how to use `CrossEntropyLoss` with `torch::nn::CrossEntropyLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(CrossEntropyLoss);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCEWithLogitsLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// This loss combines a `Sigmoid` layer and the `BCELoss` in one single
/// class. This version is more numerically stable than using a plain `Sigmoid`
/// followed by a `BCELoss` as, by combining the operations into one layer,
/// we take advantage of the log-sum-exp trick for numerical stability.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BCEWithLogitsLoss to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BCEWithLogitsLossOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BCEWithLogitsLoss model(BCEWithLogitsLossOptions().reduction(torch::kNone).weight(weight));
/// ```
struct TORCH_API BCEWithLogitsLossImpl : public Cloneable<BCEWithLogitsLossImpl> {
  explicit BCEWithLogitsLossImpl(const BCEWithLogitsLossOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `BCEWithLogitsLoss` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input, const Tensor& target);

  /// The options with which this `Module` was constructed.
  BCEWithLogitsLossOptions options;

  /// A manual rescaling weight given to the loss of each batch element.
  Tensor weight;

  /// A weight of positive examples.
  Tensor pos_weight;
};

/// A `ModuleHolder` subclass for `BCEWithLogitsLossImpl`.
/// See the documentation for `BCEWithLogitsLossImpl` class to learn what methods it
/// provides, and examples of how to use `BCEWithLogitsLoss` with `torch::nn::BCEWithLogitsLossOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BCEWithLogitsLoss);

} // namespace nn
} // namespace torch
