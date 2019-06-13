#pragma once

#include <cstdint>

namespace caffe2 {

/**
 * Embedding lookup with reduction.
 *
 * `input` of size data_size * block_size
 * `indices` of size index_size
 * `lengths` of size output_size
 * `weights` nullptr or array of size index_size
 * `out` of size output_size * block_size
 * sum(lengths[i]) == index_size
 *
 * Behavior is roughly equivalent to pseudocode:
 *
 * pos = 0
 * for (i = 0..index_size-1)
 *   for (k = 0..block_size-1)
 *     out[i*block_size + k] = 0
 *   for (j = 0..lengths[i]-1)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] += input[indices[pos]*block_size + k] *
 *           (weights ? weights[IS_WEIGHT_POSITIONAL ? j : pos] : 1.0)
 *     pos += 1
 *   if (normalize_weights && lengths[i] > 0)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] /= lengths[i]
 *
 * TODO: make this API also take "offsets" rather than "lengths" to match the
 *       API for PyTorch's EmbeddingBag
 */
template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
void EmbeddingLookup(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    const float* scale_bias, // optional scale & bias params for uint8 input
    bool normalize_by_lengths,
    OutType* out);

} // namespace caffe2
