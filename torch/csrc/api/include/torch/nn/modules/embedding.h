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
  EmbeddingOptions(int64_t count, int64_t dimension);
  /// The number of embeddings (number of rows in the table).
  TORCH_ARG(int64_t, count);
  /// The size of each embedding vector (number of columns in the table).
  TORCH_ARG(int64_t, dimension);
};

/// Performs a lookup in a fixed size embedding table.
class TORCH_API EmbeddingImpl : public torch::nn::Cloneable<EmbeddingImpl> {
 public:
  EmbeddingImpl(int64_t count, int64_t dimension)
      : EmbeddingImpl(EmbeddingOptions(count, dimension)) {}
  explicit EmbeddingImpl(EmbeddingOptions options);

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
TORCH_MODULE(Embedding);

} // namespace nn
} // namespace torch
