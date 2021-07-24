#include "caffe2/perfkernels/embedding_lookup.h"

#include "caffe2/core/types.h"
#include "caffe2/perfkernels/common.h"

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
static bool EmbeddingLookupGenericSlow(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for sum reducer
    const float* scale_bias, // optional scale & bias params for uint8 input
    bool normalize_by_lengths,
    OutType* out) {
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
        __builtin_prefetch(input + block_size * indices[current + 1], 0, 1);
      }
#endif // __GNUC__

      float w = 1.f, b = 0.f;
      if (weights) {
        w = weights[IS_WEIGHT_POSITIONAL ? i : current];
      }
      if (scale_bias) {
        b = w * scale_bias[2 * indices[current] + 1];
        w = w * scale_bias[2 * indices[current]];
      }

      for (const auto j : c10::irange(block_size)) {
        out[j] += w * input[block_size * indices[current] + j] + b;
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

// Proxy back to generic implementation
#define EMBEDDING_SPECIALIZATION(                                                                  \
    IndexType, InTypeName, InType, OutType, IS_WEIGHT_POSITIONAL)                                  \
  bool                                                                                             \
      EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__base(     \
          const int64_t block_size,                                                                \
          const int64_t output_size,                                                               \
          const int64_t index_size,                                                                \
          const int64_t data_size,                                                                 \
          const InType* input,                                                                     \
          const IndexType* indices,                                                                \
          const int* lengths,                                                                      \
          const float* weights,                                                                    \
          const float* scale_bias,                                                                 \
          bool normalize_by_lengths,                                                               \
          OutType* out) {                                                                          \
    return EmbeddingLookupGenericSlow<                                                             \
        IndexType,                                                                                 \
        InType,                                                                                    \
        OutType,                                                                                   \
        IS_WEIGHT_POSITIONAL>(                                                                     \
        block_size,                                                                                \
        output_size,                                                                               \
        index_size,                                                                                \
        data_size,                                                                                 \
        input,                                                                                     \
        indices,                                                                                   \
        lengths,                                                                                   \
        weights,                                                                                   \
        scale_bias,                                                                                \
        normalize_by_lengths,                                                                      \
        out);                                                                                      \
  }                                                                                                \
  decltype(                                                                                        \
      EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__base)     \
      EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__avx2_fma; \
  bool                                                                                             \
      EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL(             \
          const int64_t block_size,                                                                \
          const int64_t output_size,                                                               \
          const int64_t index_size,                                                                \
          const int64_t data_size,                                                                 \
          const InType* input,                                                                     \
          const IndexType* indices,                                                                \
          const int* lengths,                                                                      \
          const float* weights,                                                                    \
          const float* scale_bias,                                                                 \
          bool normalize_by_lengths,                                                               \
          OutType* out) {                                                                          \
    if (std::is_same<InType, uint8_t>::value) {                                                    \
      CAFFE_ENFORCE(scale_bias != nullptr, "scale_bias must not be nullptr");                      \
    } else {                                                                                       \
      CAFFE_ENFORCE(scale_bias == nullptr, "scale_bias must be nullptr");                          \
    }                                                                                              \
    AVX2_FMA_DO(                                                                                   \
        EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL,           \
        block_size,                                                                                \
        output_size,                                                                               \
        index_size,                                                                                \
        data_size,                                                                                 \
        input,                                                                                     \
        indices,                                                                                   \
        lengths,                                                                                   \
        weights,                                                                                   \
        scale_bias,                                                                                \
        normalize_by_lengths,                                                                      \
        out);                                                                                      \
    BASE_DO(                                                                                       \
        EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL,           \
        block_size,                                                                                \
        output_size,                                                                               \
        index_size,                                                                                \
        data_size,                                                                                 \
        input,                                                                                     \
        indices,                                                                                   \
        lengths,                                                                                   \
        weights,                                                                                   \
        scale_bias,                                                                                \
        normalize_by_lengths,                                                                      \
        out);                                                                                      \
  }                                                                                                \
  template <>                                                                                      \
  void EmbeddingLookup<IndexType, InType, OutType, IS_WEIGHT_POSITIONAL>(                          \
      const int64_t block_size,                                                                    \
      const int64_t output_size,                                                                   \
      const int64_t index_size,                                                                    \
      const int64_t data_size,                                                                     \
      const InType* input,                                                                         \
      const IndexType* indices,                                                                    \
      const int* lengths,                                                                          \
      const float* weights,                                                                        \
      const float* scale_bias,                                                                     \
      bool normalize_by_lengths,                                                                   \
      OutType* out) {                                                                              \
    bool success =                                                                                 \
        EmbeddingLookup_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL(           \
            block_size,                                                                            \
            output_size,                                                                           \
            index_size,                                                                            \
            data_size,                                                                             \
            input,                                                                                 \
            indices,                                                                               \
            lengths,                                                                               \
            weights,                                                                               \
            scale_bias,                                                                            \
            normalize_by_lengths,                                                                  \
            out);                                                                                  \
    if (success) {                                                                                 \
      return;                                                                                      \
    }                                                                                              \
    int64_t current = 0;                                                                           \
    for (int m = 0; m < output_size; ++m) {                                                        \
      for (int i = 0; i < lengths[m]; ++i) {                                                       \
        CAFFE_ENFORCE_LT(current, index_size);                                                     \
        IndexType idx = indices[current];                                                          \
        CAFFE_ENFORCE(                                                                             \
            0 <= idx && idx < data_size,                                                           \
            "Index ",                                                                              \
            current,                                                                               \
            " is out of bounds: ",                                                                 \
            idx,                                                                                   \
            ", range 0 to ",                                                                       \
            data_size);                                                                            \
        ++current;                                                                                 \
      }                                                                                            \
    }                                                                                              \
    CAFFE_ENFORCE_EQ(                                                                              \
        current,                                                                                   \
        index_size,                                                                                \
        "Your input seems to be incorrect: the sum of lengths values should be "                   \
        "the size of the indices tensor, but it appears not.");                                    \
  }

EMBEDDING_SPECIALIZATION(int32_t, float, float, float, false);
EMBEDDING_SPECIALIZATION(int64_t, float, float, float, false);
EMBEDDING_SPECIALIZATION(int32_t, half, at::Half, float, false);
EMBEDDING_SPECIALIZATION(int64_t, half, at::Half, float, false);
EMBEDDING_SPECIALIZATION(int32_t, uint8_t, uint8_t, float, false);
EMBEDDING_SPECIALIZATION(int64_t, uint8_t, uint8_t, float, false);

EMBEDDING_SPECIALIZATION(int32_t, float, float, float, true);
EMBEDDING_SPECIALIZATION(int64_t, float, float, float, true);
EMBEDDING_SPECIALIZATION(int32_t, half, at::Half, float, true);
EMBEDDING_SPECIALIZATION(int64_t, half, at::Half, float, true);
EMBEDDING_SPECIALIZATION(int32_t, uint8_t, uint8_t, float, true);
EMBEDDING_SPECIALIZATION(int64_t, uint8_t, uint8_t, float, true);

#undef EMBEDDING_SPECIALIZATION

} // namespace caffe2
