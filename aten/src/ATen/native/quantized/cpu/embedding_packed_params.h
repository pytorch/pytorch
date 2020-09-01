#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

struct EmbeddingPackedParamsBase : public torch::jit::CustomClassHolder {
  virtual at::Tensor embeddingbag_byte(
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_,
    bool include_last_offset) = 0;

  virtual at::Tensor unpack() = 0;

  virtual int64_t bit_rate() const = 0;
  virtual int64_t version() const = 0;
};
