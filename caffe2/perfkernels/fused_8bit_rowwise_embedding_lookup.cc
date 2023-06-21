#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"

#include "caffe2/core/types.h"
#include "caffe2/perfkernels/common.h"
#include "caffe2/utils/cpuid.h"

#include <c10/util/irange.h>

namespace caffe2 {

/**
 * Base implementation does runtime dispatch for each segment of reduction
 * @return false if there is an out-of-bound error
 */
template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
static bool Fused8BitRowwiseEmbeddingLookupGenericSlow(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for sum reducer
    bool normalize_by_lengths,
    OutType* out) {
  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 8 / sizeof(InType);
  const int64_t fused_block_size = block_size + scale_bias_offset;
  int64_t current = 0;
  for (const auto m : c10::irange(output_size)) {
    memset(out, 0, sizeof(OutType) * block_size);
    if (current + lengths[m] > index_size) {
      return false;
    }
    for (int i = 0; i < lengths[m]; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }
#ifdef __GNUC__
      if (current + 1 < index_size) {
        __builtin_prefetch(
            input + fused_block_size * indices[current + 1], 0, 1);
      }
#endif // __GNUC__

      const float* scale_bias = reinterpret_cast<const float*>(
          input + fused_block_size * indices[current] + block_size);

      float weight = 1.0f;
      if (weights) {
        weight = weights[IS_WEIGHT_POSITIONAL ? i : current];
      }
      const float scale = weight * scale_bias[0];
      const float bias = weight * scale_bias[1];

      for (const auto j : c10::irange(block_size)) {
        out[j] += scale * input[fused_block_size * indices[current] + j] + bias;
      }

      ++current;
    }
    if (normalize_by_lengths && lengths[m]) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      float scale = 1.f / lengths[m];
      for (const auto j : c10::irange(block_size)) {
        out[j] *= scale;
      }
    }
    out += block_size;
  }
  return current == index_size;
}

// clang-format off
// Proxy back to generic implementation
#define FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(IndexType, OutType)                  \
  static bool                                                                                   \
      Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__base(     \
          const int64_t block_size,                                                      \
          const int64_t output_size,                                                     \
          const int64_t index_size,                                                      \
          const int64_t data_size,                                                       \
          const uint8_t* input,                                                          \
          const IndexType* indices,                                                      \
          const int* lengths,                                                            \
          const float* weights,                                                          \
          bool normalize_by_lengths,                                                     \
          OutType* out) {                                                                \
    return Fused8BitRowwiseEmbeddingLookupGenericSlow<                                   \
        IndexType,                                                                       \
        uint8_t,                                                                         \
        OutType,                                                                         \
        false>(                                                                          \
        block_size,                                                                      \
        output_size,                                                                     \
        index_size,                                                                      \
        data_size,                                                                       \
        input,                                                                           \
        indices,                                                                         \
        lengths,                                                                         \
        weights,                                                                         \
        normalize_by_lengths,                                                            \
        out);                                                                            \
  }                                                                                      \
  decltype(                                                                              \
      Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__base)     \
      Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__avx2_fma; \
  static bool Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType(                  \
      const int64_t block_size,                                                          \
      const int64_t output_size,                                                         \
      const int64_t index_size,                                                          \
      const int64_t data_size,                                                           \
      const uint8_t* input,                                                              \
      const IndexType* indices,                                                          \
      const int* lengths,                                                                \
      const float* weights,                                                              \
      bool normalize_by_lengths,                                                         \
      OutType* out) {                                                                    \
    const int32_t one = 1;                                                               \
    CAFFE_ENFORCE_EQ(                                                                    \
        reinterpret_cast<const uint8_t*>(&one)[0],                                       \
        1,                                                                               \
        "Fused8BitRowwiseEmbeddingLookup is not supported on this platform");            \
    AVX2_FMA_DO(                                                                         \
        Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false,         \
        block_size,                                                                      \
        output_size,                                                                     \
        index_size,                                                                      \
        data_size,                                                                       \
        input,                                                                           \
        indices,                                                                         \
        lengths,                                                                         \
        weights,                                                                         \
        normalize_by_lengths,                                                            \
        out);                                                                            \
    BASE_DO(                                                                             \
        Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false,         \
        block_size,                                                                      \
        output_size,                                                                     \
        index_size,                                                                      \
        data_size,                                                                       \
        input,                                                                           \
        indices,                                                                         \
        lengths,                                                                         \
        weights,                                                                         \
        normalize_by_lengths,                                                            \
        out);                                                                            \
  }                                                                                      \
  template <>                                                                            \
  void Fused8BitRowwiseEmbeddingLookup<IndexType, uint8_t, OutType, false>(              \
      const int64_t block_size,                                                          \
      const int64_t output_size,                                                         \
      const int64_t index_size,                                                          \
      const int64_t data_size,                                                           \
      const uint8_t* input,                                                              \
      const IndexType* indices,                                                          \
      const int* lengths,                                                                \
      const float* weights,                                                              \
      bool normalize_by_lengths,                                                         \
      OutType* out) {                                                                    \
    bool success =                                                                       \
        Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType(                 \
            block_size,                                                                  \
            output_size,                                                                 \
            index_size,                                                                  \
            data_size,                                                                   \
            input,                                                                       \
            indices,                                                                     \
            lengths,                                                                     \
            weights,                                                                     \
            normalize_by_lengths,                                                        \
            out);                                                                        \
    if (success) {                                                                       \
      return;                                                                            \
    }                                                                                    \
    int64_t current = 0;                                                                 \
    for (int m = 0; m < output_size; ++m) {                                              \
      for (int i = 0; i < lengths[m]; ++i) {                                             \
        CAFFE_ENFORCE_LT(current, index_size);                                           \
        IndexType idx = indices[current];                                                \
        CAFFE_ENFORCE(                                                                   \
            0 <= idx && idx < data_size,                                                 \
            "Index ",                                                                    \
            current,                                                                     \
            " is out of bounds: ",                                                       \
            idx,                                                                         \
            ", range 0 to ",                                                             \
            data_size);                                                                  \
        ++current;                                                                       \
      }                                                                                  \
    }                                                                                    \
    CAFFE_ENFORCE_EQ(                                                                    \
        current,                                                                         \
        index_size,                                                                      \
        "Your input seems to be incorrect: the sum of lengths values should be "         \
        "the size of the indices tensor, but it appears not.");                          \
  }
// clang-format on

FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(int32_t, float);
FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(int64_t, float);

#undef FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION

} // namespace caffe2
