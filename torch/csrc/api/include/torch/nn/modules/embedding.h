#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>

namespace torch {
namespace nn {

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

  static Embedding from_pretrained(const torch::Tensor& embeddings, EmbeddingOptions options = {}, bool freeze = true) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");
    if (options.num_embeddings()) {
      TORCH_WARN("`num_embeddings` options parameter is ignored in `torch::nn::Embedding::from_pretrained`.");
    }
    if (options.embedding_dim()) {
      TORCH_WARN("`embedding_dim` options parameter is ignored in `torch::nn::Embedding::from_pretrained`.");
    }

    Embedding embedding(options.num_embeddings(embeddings.size(0)).embedding_dim(embeddings.size(1))._weight(embeddings));
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

  Tensor forward(const Tensor& input, const Tensor& offsets = {}, const Tensor& per_sample_weights = {});

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

  static EmbeddingBag from_pretrained(const torch::Tensor& embeddings, EmbeddingBagOptions options = {}, bool freeze = true) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");
    if (options.num_embeddings()) {
      TORCH_WARN("`num_embeddings` options parameter is ignored in `torch::nn::EmbeddingBag::from_pretrained`.");
    }
    if (options.embedding_dim()) {
      TORCH_WARN("`embedding_dim` options parameter is ignored in `torch::nn::EmbeddingBag::from_pretrained`.");
    }
    EmbeddingBag embeddingbag(options.num_embeddings(embeddings.size(0)).embedding_dim(embeddings.size(1))._weight(embeddings));
    embeddingbag->weight.set_requires_grad(!freeze);
    return embeddingbag;
  }
};
} // namespace nn
} // namespace torch
