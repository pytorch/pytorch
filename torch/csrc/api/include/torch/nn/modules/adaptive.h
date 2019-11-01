#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/options/adaptive.h>

namespace torch {
namespace nn {
  /// Classifier list for least frequent labels

/// The output of a single invocation of an AdaptiveLogSoftmaxWithLoss
/// module's `forward()` method.
struct TORCH_API ASMoutput {
  ASMoutput(const Tensor& output_, const double& loss_);
  // Tensor containing computed target log probabilities for each example
  Tensor output;

  //Scalar representing the computed negative log likelihood loss
  double loss;
};

/// Efficient softmax approximation as described in
/// Efficient softmax approximation for GPUs`_ by Edouard Grave, Armand Joulin,
/// Moustapha Cissé, David Grangier, and Hervé Jégou.
///
/// Adaptive softmax is an approximate strategy for training models with large
/// output spaces. It is most effective when the label distribution is highly
/// imbalanced, for example in natural language modelling, where the word
/// frequency distribution approximately follows the `Zipf's law`_.
///
/// Adaptive softmax partitions the labels into several clusters, according to
/// their frequency. These clusters may contain different number of targets
/// each.
/// Additionally, clusters containing less frequent labels assign lower
/// dimensional embeddings to those labels, which speeds up the computation.
/// For each minibatch, only clusters for which at least one target is
/// present are evaluated.
///
/// The idea is that the clusters which are accessed frequently
/// (like the first one, containing most frequent labels), should also be cheap
/// to compute -- that is, contain a small number of assigned labels.
///
/// We highly recommend taking a look at the original paper for more details.
/// _Efficient softmax approximation for GPUs:
/// https://arxiv.org/abs/1609.04309
///
/// _Zipf's law:
/// https://en.wikipedia.org/wiki/Zipf%27s_law

class TORCH_API AdaptiveLogSoftmaxWithLossImpl : public Cloneable<AdaptiveLogSoftmaxWithLossImpl> {
 public:
   AdaptiveLogSoftmaxWithLossImpl(int64_t in_features, int64_t n_classes, std::vector<int64_t> cutoffs)
      : AdaptiveLogSoftmaxWithLossImpl(AdaptiveLogSoftmaxWithLossOptions(in_features, n_classes, cutoffs)) {}
     
  explicit AdaptiveLogSoftmaxWithLossImpl(const AdaptiveLogSoftmaxWithLossOptions& options_);

  ASMoutput forward(const Tensor& input, const Tensor& target);

  void reset() override;

  /// Pretty prints the `LocalResponseNormImpl` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Given input tensor, and output of `head`, computes the log of the full distribution
  Tensor _get_full_log_prob(const Tensor &input, const Tensor& head_output);

  /// Computes log probabilities for all n_classes
  Tensor log_prob(const Tensor& input);

  /// This is equivalent to `log_pob(input).argmax(1)` but is more efficient in some cases
  Tensor predict(const Tensor& input);

  /// The options with which this `Module` was constructed
  AdaptiveLogSoftmaxWithLossOptions options;

  /// Cutoffs used to assign targets to their buckets. It should be an ordered Sequence 
  /// of integers sorted in the increasing order
  std::vector<int64_t> cutoffs;

  int64_t shortlist_size;
  
  /// Number of clusters
  int64_t n_clusters;

  /// Output size of head classifier
  int64_t head_size;

  Linear head = nullptr;

  ModuleList tail = nullptr;
};

TORCH_MODULE(AdaptiveLogSoftmaxWithLoss);

} // namespace nn
} // namespace torc