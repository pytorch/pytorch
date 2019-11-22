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

  static Embedding from_pretrained(const torch::Tensor& embeddings, const EmbeddingFromPretrainedOptions& options = {}) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");

    int64_t rows, cols;
    rows = embeddings.size(0);
    cols = embeddings.size(1);

    Embedding embedding(
      EmbeddingOptions(rows, cols)
        ._weight(embeddings)
        .padding_idx(options.padding_idx())
        .max_norm(options.max_norm())
        .norm_type(options.norm_type())
        .scale_grad_by_freq(options.scale_grad_by_freq())
        .sparse(options.sparse()));
    embedding->weight.set_requires_grad(!options.freeze());
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

  static EmbeddingBag from_pretrained(const torch::Tensor& embeddings, const EmbeddingBagFromPretrainedOptions& options = {}) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");

    int64_t rows, cols;
    rows = embeddings.size(0);
    cols = embeddings.size(1);

    EmbeddingBag embeddingbag(
      EmbeddingBagOptions(rows, cols)
        ._weight(embeddings)
        .max_norm(options.max_norm())
        .norm_type(options.norm_type())
        .scale_grad_by_freq(options.scale_grad_by_freq())
        .mode(options.mode())
        .sparse(options.sparse()));
    embeddingbag->weight.set_requires_grad(!options.freeze());
    return embeddingbag;
  }
};
} // namespace nn
} // namespace torch
