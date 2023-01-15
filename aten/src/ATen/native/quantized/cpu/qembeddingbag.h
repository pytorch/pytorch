#pragma once
#include <ATen/core/Tensor.h>
#include <cstdint>

namespace at {
namespace native {
Tensor& embedding_bag_byte_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset);

Tensor& embedding_bag_4bit_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset);

Tensor& qembeddingbag_byte_unpack_out(Tensor& output, const Tensor& packed_weight);

} // native
} // at
