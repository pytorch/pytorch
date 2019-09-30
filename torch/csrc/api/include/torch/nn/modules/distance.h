#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/distance.h>
#include <torch/nn/options/distance.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Returns the cosine similarity between :math:`x_1` and :math:`x_2`, computed
/// along `dim`.
class TORCH_API CosineSimilarityImpl : public Cloneable<CosineSimilarityImpl> {
 public:
  explicit CosineSimilarityImpl(const CosineSimilarityOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `CosineSimilarity` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  CosineSimilarityOptions options;
};

/// A `ModuleHolder` subclass for `CosineSimilarityImpl`.
/// See the documentation for `CosineSimilarityImpl` class to learn what methods
/// it provides, or the documentation for `ModuleHolder` to learn about
/// Pytorch's module storage semantics.
TORCH_MODULE(CosineSimilarity);

// ============================================================================

/// Returns the batchwise pairwise distance between vectors :math:`v_1`,
/// :math:`v_2` using the p-norm.
class TORCH_API PairwiseDistanceImpl : public Cloneable<PairwiseDistanceImpl> {
 public:
  explicit PairwiseDistanceImpl(const PairwiseDistanceOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `PairwiseDistance` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options with which this `Module` was constructed.
  PairwiseDistanceOptions options;
};

/// A `ModuleHolder` subclass for `PairwiseDistanceImpl`.
/// See the documentation for `PairwiseDistanceImpl` class to learn what methods
/// it provides, or the documentation for `ModuleHolder` to learn about
/// Pytorch's module storage semantics.
TORCH_MODULE(PairwiseDistance);

} // namespace nn
} // namespace torch
