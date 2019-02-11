#pragma once

#include <cstdint>

namespace caffe2 {

/**
 * Embedding lookup with reduction.
 *
 * `input` of size data_size * (block_size + 8B)
 * `indices` of size index_size
 * `lengths` of size output_size
 * `weights` nullptr or array of size index_size
 * `out` of size output_size * block_size
 * sum(lengths[i]) == index_size
 *
 * Note that block_size should be the number of quantized values per row in the
 * data, i.e. excluding the scale and bias. The total (fused) block size is
 * assumed to be this block_size, plus 4 bytes for scale and 4 bytes for bias.
 *
 * Behavior is roughly equivalent to pseudocode:
 *
 * pos = 0
 * fused_block_size = block_size + 8B // quantized values and scale and bias
 * for (i = 0..index_size-1)
 *   for (k = 0..block_size-1)
 *     out[i*block_size + k] = 0
 *   for (j = 0..lengths[i]-1)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] += input[indices[pos]*(fused_block_size) + k] *
 *           (weights ? weights[IS_WEIGHT_POSITIONAL ? j : pos] : 1.0)
 *     pos += 1
 *   if (normalize_weights && lengths[i] > 0)
 *     for (k = 0..block_size-1)
 *       out[i*block_size + k] /= lengths[i]
 *
 */

template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
void Fused8BitRowwiseEmbeddingLookup(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    OutType* out);
} // namespace caffe2
