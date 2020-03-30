#pragma once

#include <torch/nn/options/embedding.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor one_hot(const Tensor& tensor, int64_t num_classes = -1) {
  return torch::one_hot(tensor, num_classes);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline void _no_grad_embedding_renorm_(Tensor weight, const Tensor& input, float max_norm, float norm_type) {
  torch::NoGradGuard no_grad;
  torch::embedding_renorm_(weight, input, max_norm, norm_type);
}

inline Tensor embedding(const Tensor& input,
                        const Tensor& weight,
                        c10::optional<int64_t> padding_idx,
                        c10::optional<double> max_norm,
                        double norm_type,
                        bool scale_grad_by_freq,
                        bool sparse) {
  auto input_ = input;

  if (padding_idx != c10::nullopt) {
    if (*padding_idx > 0) {
      TORCH_CHECK(*padding_idx < weight.size(0), "Padding_idx must be within num_embeddings");
    }
    else if (*padding_idx < 0) {
      TORCH_CHECK(*padding_idx >= -weight.size(0), "Padding_idx must be within num_embedding");
      padding_idx = weight.size(0) + *padding_idx;
    }
  } else {
    padding_idx = -1;
  }

  if (max_norm != c10::nullopt) {
    input_ = input_.contiguous();
    _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
  }
  return torch::embedding(weight, input_, *padding_idx, scale_grad_by_freq, sparse);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.embedding
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::EmbeddingFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding(input, weight, F::EmbeddingFuncOptions().norm_type(2.5).scale_grad_by_freq(true).sparse(true));
/// ```
inline Tensor embedding(const Tensor& input, const Tensor& weight, const EmbeddingFuncOptions& options = {}) {
  return detail::embedding(
    input,
    weight,
    options.padding_idx(),
    options.max_norm(),
    options.norm_type(),
    options.scale_grad_by_freq(),
    options.sparse());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor embedding_bag(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offsets,
    c10::optional<double> max_norm,
    double norm_type,
    bool scale_grad_by_freq,
    EmbeddingBagMode mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset) {
  auto input_ = input;
  auto offsets_ = offsets;
  auto per_sample_weights_ = per_sample_weights;
  TORCH_CHECK(!per_sample_weights_.defined() || input_.sizes() == per_sample_weights_.sizes(),
    "embedding_bag: If per_sample_weights (", per_sample_weights_.sizes(), ") is not null, then it must have the same shape as the input (", input_.sizes(), ")");
  if (input_.dim() == 2) {
    TORCH_CHECK(!offsets_.defined(),
      "If input is 2D, then offsets has to be null, as input is treated is a mini-batch of fixed length sequences. However, found offsets of type Tensor");
    offsets_ = torch::arange(0, input_.numel(), input_.size(1), torch::TensorOptions().dtype(torch::kLong).device(input_.device()));
    input_ = input_.reshape(-1);
    if (per_sample_weights_.defined()) {
      per_sample_weights_ = per_sample_weights_.reshape(-1);
    }
  } else if (input_.dim() == 1) {
    TORCH_CHECK(offsets_.defined(), "offsets has to be a 1D Tensor but got null");
    TORCH_CHECK(offsets_.dim() == 1, "offsets has to be a 1D Tensor");
    TORCH_CHECK(offsets_[0].item<int64_t>() == 0, "offsets[0] has to be 0, i.e., the first sequence in the mini-batch has to start from position 0. However, got ",
     offsets_[0].item<int64_t>());
    TORCH_CHECK(offsets_[-1].item<int64_t>() <= input_.size(0), "offsets[-1] can not be greater than input's length({",
              input_.size(0), "}), but got offsets[-1] of {", offsets_[-1].item<int64_t>(), "}");
  } else {
    TORCH_CHECK(false, "input has to be 1D or 2D Tensor, but got Tensor of dimension ", input_.dim());
  }

  int mode_enum;
  if (c10::get_if<enumtype::kSum>(&mode)) {
    mode_enum = 0;
  } else if (c10::get_if<enumtype::kMean>(&mode)) {
    mode_enum = 1;
  } else if (c10::get_if<enumtype::kMax>(&mode)) {
    mode_enum = 2;
    TORCH_CHECK(!scale_grad_by_freq, "max mode does not support scaling the gradient by the frequency");
    TORCH_CHECK(!sparse, "max mode does not support sparse weights");
  } else {
    TORCH_CHECK(false, "mode has to be one of sum, mean or max");
  }

  if (max_norm != c10::nullopt) {
    _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
  }

  TORCH_CHECK(
    !per_sample_weights_.defined() || c10::get_if<enumtype::kSum>(&mode),
    "embedding_bag: per_sample_weights was not null. ",
    "per_sample_weights is only supported for mode='kSum' (got mode='",
    torch::enumtype::get_enum_name(mode), "').Please open a feature request on GitHub.");

  return std::get<0>(
    torch::embedding_bag(
      weight,
      input_,
      offsets_,
      scale_grad_by_freq,
      mode_enum,
      sparse,
      per_sample_weights_,
      include_last_offset));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.embedding_bag
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::EmbeddingBagFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::embedding_bag(input, weight, F::EmbeddingBagFuncOptions().mode(torch::kSum).offsets(offsets));
/// ```
inline Tensor embedding_bag(const Tensor& input, const Tensor& weight, const EmbeddingBagFuncOptions& options = {}) {
  return detail::embedding_bag(
    input,
    weight,
    options.offsets(),
    options.max_norm(),
    options.norm_type(),
    options.scale_grad_by_freq(),
    options.mode(),
    options.sparse(),
    options.per_sample_weights(),
    options.include_last_offset());
}

} // namespace functional
} // namespace nn
} // namespace torch
