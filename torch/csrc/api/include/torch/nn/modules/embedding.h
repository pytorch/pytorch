#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Options for the `Embedding` module.
struct TORCH_API EmbeddingOptions {
  EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim) :
   num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {};
  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  TORCH_ARG(int64_t, embedding_dim);
  /// If given, pads the output with the embedding vector at `padding_idx` (initialized to zeros) whenever it encounters the index.
  TORCH_ARG(c10::optional<int64_t>, padding_idx) = c10::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`.
  TORCH_ARG(c10::optional<float>, max_norm) = c10::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(float, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default ``False``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  TORCH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings, embedding_dim)
  TORCH_ARG(torch::Tensor, _weight) = Tensor();
};

/// Options for the `EmbeddingBag` module.
struct TORCH_API EmbeddingBagOptions {
  EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim) :
   num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {};
  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  TORCH_ARG(int64_t, embedding_dim);
  /// If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`.
  TORCH_ARG(c10::optional<float>, max_norm) = c10::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(float, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default ``False``.
  /// Note: this option is not supported when ``mode="max"``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag. ``"sum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"mean"`` computes the average of the values in the bag, ``"max"`` computes the max value over each bag.
  TORCH_ARG(string, mode) = "mean";
  /// If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="max"``.
  TORCH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings, embedding_dim)
  TORCH_ARG(torch::Tensor, _weight) = Tensor();
};

/// Performs a lookup in a fixed size embedding table.
class TORCH_API EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
     : EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingImpl(const EmbeddingOptions& options_);

  void reset() override;

  /// Pretty prints the `Embedding` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Performs a lookup on the embedding table stored in `weight` using the
  /// `indices` supplied and returns the result.
  Tensor forward(const Tensor& indices);

  /// The `Options` used to configure this `Embedding` module.
  /// Changes to `EmbeddingOptions` *after construction* have no effect.
  EmbeddingOptions options;

  /// The embedding table.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `EmbeddingImpl`.
/// See the documentation for `EmbeddingImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
class Embedding : public torch::nn::ModuleHolder<EmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingImpl>::ModuleHolder;

  static Embedding from_pretrained(const torch::Tensor& embeddings, c10::optional<EmbeddingOptions> options = c10::nullopt, bool freeze = true) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");
    if (options != c10::nullopt) {
      TORCH_CHECK((*options).num_embeddings() == embeddings.size(0), "Expects options.num_embeddings to be ", embeddings.size(0) , "but found ", (*options).num_embeddings());
      TORCH_CHECK((*options).embedding_dim() == embeddings.size(1), "Expects options.embeddings_dim to be ", embeddings.size(1) , "but found ", (*options).embedding_dim());
    } else {
      options = EmbeddingOptions(embeddings.size(0), embeddings.size(1));
    }
    Embedding embedding((*options)._weight(embeddings));
    embedding->weight.set_requires_grad(!freeze);
    return embedding;
  }
};

class TORCH_API EmbeddingBagImpl : public torch::nn::Cloneable<EmbeddingBagImpl> {
 public:
  EmbeddingBagImpl(int64_t num_embeddings, int64_t embedding_dim)
    : EmbeddingBagImpl(EmbeddingBagOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingBagImpl(const EmbeddingBagOptions& options_);

  void reset() override;

  /// Pretty prints the `EmbeddingBag` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(const Tensor& input, const torch::Tensor& offsets = torch::Tensor(),
    const torch::Tensor& per_sample_weights = torch::Tensor());

  /// The `Options` used to configure this `EmbeddingBag` module.
  EmbeddingBagOptions options;
  /// The embedding table.
  Tensor weight;
};

/// A `ModuleHolder` subclass for `EmbeddingBagImpl`.
/// See the documentation for `EmbeddingBagImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
class EmbeddingBag : public torch::nn::ModuleHolder<EmbeddingBagImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingBagImpl>::ModuleHolder;

  static EmbeddingBag from_pretrained(const torch::Tensor& embeddings, c10::optional<EmbeddingBagOptions> options = c10::nullopt, bool freeze = true) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");
    if (options != c10::nullopt) {
      TORCH_CHECK((*options).num_embeddings() == embeddings.size(0), "Expects options.num_embeddings to be ", embeddings.size(0) , "but found ", (*options).num_embeddings());
      TORCH_CHECK((*options).embedding_dim() == embeddings.size(1), "Expects options.embeddings_dim to be ", embeddings.size(1) , "but found ", (*options).embedding_dim());
    } else {
      options = EmbeddingBagOptions(embeddings.size(0), embeddings.size(1));
    }
    EmbeddingBag embeddingbag((*options)._weight(embeddings));
    embeddingbag->weight.set_requires_grad(!freeze);
    return embeddingbag;
  }
};
} // namespace nn
} // namespace torch
