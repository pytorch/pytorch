#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <torch/enum.h>

namespace torch {
namespace nn {
  /// Options for the `Embedding` module.
  struct TORCH_API EmbeddingOptions {
    EmbeddingOptions();
    EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim);
    /// The size of the dictionary of embeddings.
    TORCH_ARG(c10::optional<int64_t>, num_embeddings) = c10::nullopt;
    /// The size of each embedding vector.
    TORCH_ARG(c10::optional<int64_t>, embedding_dim) = c10::nullopt;
    /// If given, pads the output with the embedding vector at `padding_idx` (initialized to zeros) whenever it encounters the index.
    TORCH_ARG(c10::optional<int64_t>, padding_idx) = c10::nullopt;
    /// If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`.
    TORCH_ARG(c10::optional<double>, max_norm) = c10::nullopt;
    /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
    TORCH_ARG(double, norm_type) = 2.;
    /// If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default ``False``.
    TORCH_ARG(bool, scale_grad_by_freq) = false;
    /// If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
    TORCH_ARG(bool, sparse) = false;
    /// The learnable weights of the module of shape (num_embeddings, embedding_dim)
    TORCH_ARG(torch::Tensor, _weight) = Tensor();
  };

  /// Options for the `EmbeddingBag` module.
  struct TORCH_API EmbeddingBagOptions {
    typedef c10::variant<enumtype::kSum, enumtype::kMean, enumtype::kMax> mode_t;
    EmbeddingBagOptions();
    EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim);
    /// The size of the dictionary of embeddings.
    TORCH_ARG(c10::optional<int64_t>, num_embeddings) = c10::nullopt;
    /// The size of each embedding vector.
    TORCH_ARG(c10::optional<int64_t>, embedding_dim) = c10::nullopt;
    /// If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`.
    TORCH_ARG(c10::optional<double>, max_norm) = c10::nullopt;
    /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
    TORCH_ARG(double, norm_type) = 2.;
    /// If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default ``False``.
    /// Note: this option is not supported when ``mode="kMax"``.
    TORCH_ARG(bool, scale_grad_by_freq) = false;
    /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
    /// into consideration. ``"kMean"`` computes the average of the values in the bag, ``"kMax"`` computes the max value over each bag.
    TORCH_ARG(mode_t, mode) = torch::kMean;
    /// If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor.
    /// Note: this option is not supported when ``mode="kMax"``.
    TORCH_ARG(bool, sparse) = false;
    /// The learnable weights of the module of shape (num_embeddings, embedding_dim)
    TORCH_ARG(torch::Tensor, _weight) = Tensor();
  };
} // namespace nn
} // namespace torch
