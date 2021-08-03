#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Embedding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Performs a lookup in a fixed size embedding table.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Embedding to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::EmbeddingOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
// NOLINTNEXTLINE(bugprone-exception-escape)
class TORCH_API EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  EmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim)
     : EmbeddingImpl(EmbeddingOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingImpl(const EmbeddingOptions& options_);

  void reset() override;

  void reset_parameters();

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
/// provides, and examples of how to use `Embedding` with `torch::nn::EmbeddingOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
class Embedding : public torch::nn::ModuleHolder<EmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingImpl>::ModuleHolder;

  /// See the documentation for `torch::nn::EmbeddingFromPretrainedOptions` class to learn what
  /// optional arguments are supported for this function.
  static Embedding from_pretrained(const torch::Tensor& embeddings, const EmbeddingFromPretrainedOptions& options = {}) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EmbeddingBag ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Computes sums or means of 'bags' of embeddings, without instantiating the
/// intermediate embeddings.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.EmbeddingBag to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::EmbeddingBagOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10, 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum).padding_idx(1));
/// ```
// NOLINTNEXTLINE(bugprone-exception-escape)
class TORCH_API EmbeddingBagImpl : public torch::nn::Cloneable<EmbeddingBagImpl> {
 public:
  EmbeddingBagImpl(int64_t num_embeddings, int64_t embedding_dim)
    : EmbeddingBagImpl(EmbeddingBagOptions(num_embeddings, embedding_dim)) {}
  explicit EmbeddingBagImpl(const EmbeddingBagOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `EmbeddingBag` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The `Options` used to configure this `EmbeddingBag` module.
  EmbeddingBagOptions options;
  /// The embedding table.
  Tensor weight;

  Tensor forward(const Tensor& input, const Tensor& offsets = {}, const Tensor& per_sample_weights = {});
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `EmbeddingBagImpl`.
/// See the documentation for `EmbeddingBagImpl` class to learn what methods it
/// provides, and examples of how to use `EmbeddingBag` with `torch::nn::EmbeddingBagOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
class EmbeddingBag : public torch::nn::ModuleHolder<EmbeddingBagImpl> {
 public:
  using torch::nn::ModuleHolder<EmbeddingBagImpl>::ModuleHolder;

  /// See the documentation for `torch::nn::EmbeddingBagFromPretrainedOptions` class to learn what
  /// optional arguments are supported for this function.
  static EmbeddingBag from_pretrained(const torch::Tensor& embeddings, const EmbeddingBagFromPretrainedOptions& options = {}) {
    TORCH_CHECK(embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional");

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
        .sparse(options.sparse())
        .padding_idx(options.padding_idx()));
    embeddingbag->weight.set_requires_grad(!options.freeze());
    return embeddingbag;
  }
};
} // namespace nn
} // namespace torch
