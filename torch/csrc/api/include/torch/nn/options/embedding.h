#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for the `Embedding` module.
///
/// Example:
/// ```
/// Embedding model(EmbeddingOptions(10,
/// 2).padding_idx(3).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
struct TORCH_API EmbeddingOptions {
  EmbeddingOptions(int64_t num_embeddings, int64_t embedding_dim);

  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  TORCH_ARG(int64_t, embedding_dim);
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad". For a newly constructed
  /// Embedding, the embedding vector at `padding_idx` will default to all
  /// zeros, but can be updated to another value to be used as the padding
  /// vector.
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  TORCH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings,
  /// embedding_dim)
  TORCH_ARG(torch::Tensor, _weight) = Tensor();
};

// ============================================================================

/// Options for the `Embedding::from_pretrained` function.
struct TORCH_API EmbeddingFromPretrainedOptions {
  /// If ``true``, the tensor does not get updated in the learning process.
  /// Equivalent to ``embedding.weight.requires_grad_(false)``. Default:
  /// ``true``
  TORCH_ARG(bool, freeze) = true;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad".
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  TORCH_ARG(bool, sparse) = false;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::embedding`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding(input, weight,
/// F::EmbeddingFuncOptions().norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
struct TORCH_API EmbeddingFuncOptions {
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at `padding_idx` is not updated
  /// during training, i.e. it remains as a fixed "pad".
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  TORCH_ARG(bool, sparse) = false;
};

} // namespace functional

// ============================================================================

typedef std::variant<enumtype::kSum, enumtype::kMean, enumtype::kMax>
    EmbeddingBagMode;

/// Options for the `EmbeddingBag` module.
///
/// Example:
/// ```
/// EmbeddingBag model(EmbeddingBagOptions(10,
/// 2).max_norm(2).norm_type(2.5).scale_grad_by_freq(true).sparse(true).mode(torch::kSum));
/// ```
struct TORCH_API EmbeddingBagOptions {
  EmbeddingBagOptions(int64_t num_embeddings, int64_t embedding_dim);

  /// The size of the dictionary of embeddings.
  TORCH_ARG(int64_t, num_embeddings);
  /// The size of each embedding vector.
  TORCH_ARG(int64_t, embedding_dim);
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  TORCH_ARG(bool, sparse) = false;
  /// The learnable weights of the module of shape (num_embeddings,
  /// embedding_dim)
  TORCH_ARG(torch::Tensor, _weight) = Tensor();
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format.
  TORCH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". For a newly constructed
  /// EmbeddingBag, the embedding vector at `padding_idx` will default to all
  /// zeros, but can be updated to another value to be used as the padding
  /// vector. Note that the embedding vector at `padding_idx` is excluded from
  /// the reduction.
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

// ============================================================================

/// Options for the `EmbeddingBag::from_pretrained` function.
struct TORCH_API EmbeddingBagFromPretrainedOptions {
  /// If ``true``, the tensor does not get updated in the learning process.
  /// Equivalent to ``embeddingbag.weight.requires_grad_(false)``. Default:
  /// ``true``
  TORCH_ARG(bool, freeze) = true;
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  TORCH_ARG(bool, sparse) = false;
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format. Note:
  /// this option is currently only supported when ``mode="sum"``.
  TORCH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". Note that the embedding
  /// vector at `padding_idx` is excluded from the reduction.
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

// ============================================================================

namespace functional {

/// Options for `torch::nn::functional::embedding_bag`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding_bag(input, weight,
/// F::EmbeddingBagFuncOptions().mode(torch::kSum).offsets(offsets));
/// ```
struct TORCH_API EmbeddingBagFuncOptions {
  /// Only used when `input` is 1D. `offsets` determines
  /// the starting index position of each bag (sequence) in `input`.
  TORCH_ARG(torch::Tensor, offsets) = Tensor();
  /// If given, each embedding vector with norm larger than `max_norm` is
  /// renormalized to have norm `max_norm`.
  TORCH_ARG(std::optional<double>, max_norm) = std::nullopt;
  /// The p of the p-norm to compute for the `max_norm` option. Default ``2``.
  TORCH_ARG(double, norm_type) = 2.;
  /// If given, this will scale gradients by the inverse of frequency of the
  /// words in the mini-batch. Default ``false``. Note: this option is not
  /// supported when ``mode="kMax"``.
  TORCH_ARG(bool, scale_grad_by_freq) = false;
  /// ``"kSum"``, ``"kMean"`` or ``"kMax"``. Specifies the way to reduce the
  /// bag. ``"kSum"`` computes the weighted sum, taking `per_sample_weights`
  /// into consideration. ``"kMean"`` computes the average of the values in the
  /// bag, ``"kMax"`` computes the max value over each bag.
  TORCH_ARG(EmbeddingBagMode, mode) = torch::kMean;
  /// If ``true``, gradient w.r.t. `weight` matrix will be a sparse tensor.
  /// Note: this option is not supported when ``mode="kMax"``.
  TORCH_ARG(bool, sparse) = false;
  /// a tensor of float / double weights, or None to indicate all weights should
  /// be taken to be 1. If specified, `per_sample_weights` must have exactly the
  /// same shape as input and is treated as having the same `offsets`, if those
  /// are not None.
  TORCH_ARG(torch::Tensor, per_sample_weights) = Tensor();
  /// If ``true``, `offsets` has one additional element, where the last element
  /// is equivalent to the size of `indices`. This matches the CSR format. Note:
  /// this option is currently only supported when ``mode="sum"``.
  TORCH_ARG(bool, include_last_offset) = false;
  /// If specified, the entries at `padding_idx` do not contribute to the
  /// gradient; therefore, the embedding vector at padding_idx is not updated
  /// during training, i.e. it remains as a fixed "pad". Note that the embedding
  /// vector at `padding_idx` is excluded from the reduction.
  TORCH_ARG(std::optional<int64_t>, padding_idx) = std::nullopt;
};

} // namespace functional

} // namespace nn
} // namespace torch
