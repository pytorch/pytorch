#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

struct EmbeddingPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor embeddingbag_byte(
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) = 0;

  virtual at::Tensor embeddingbag_4bit(
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) = 0;

  virtual at::Tensor unpack() = 0;

  virtual int64_t bit_rate() const = 0;
  virtual int64_t version() const = 0;
};
