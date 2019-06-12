#include "caffe2/perfkernels/embedding_lookup.h"

#include "caffe2/core/types.h"
#include "caffe2/perfkernels/common.h"
#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Base implementation does runtime dispatch for each segment of reduction
template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
static void EmbeddingLookupGenericSlow(
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
  for (int m = 0; m < output_size; ++m) {
    memset(out, 0, sizeof(OutType) * block_size);
    EigenVectorArrayMap<OutType> out_vector(out, block_size);
    for (int i = 0; i < lengths[m]; ++i) {
      CAFFE_ENFORCE_LT(current, index_size);
      int64_t idx = indices[current];
      CAFFE_ENFORCE(
          0 <= idx && idx < data_size,
          "Index ",
          current,
          " is out of bounds: ",
          idx,
          ", range 0 to ",
          data_size);
      CAFFE_ENFORCE_LT(idx, data_size);
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

      TypedAxpy<InType, OutType>(
          block_size, w, input + block_size * indices[current], out);

      if (scale_bias) {
        out_vector = out_vector + b;
      }

      ++current;
    }
    if (normalize_by_lengths && lengths[m]) {
      // hack: context is not really used
      math::Scale<float, OutType, CPUContext>(
          block_size, 1.f / lengths[m], out, out, nullptr);
    }
    out += block_size;
  }
  CAFFE_ENFORCE_EQ(
      current,
      index_size,
      "Your input seems to be incorrect: the sum of lengths values should be "
      "the size of the indices tensor, but it appears not.");
}

// Proxy back to generic implementation
#define EMBEDDING_SPECIALIZATION(                                                                      \
    IndexTypeName, IndexType, InTypeName, InType, OutTypeName, OutType, IS_WEIGHT_POSITIONAL)          \
  void                                                                                                 \
      EmbeddingLookup_##IndexTypeName##_##InTypeName##_##OutTypeName##_##IS_WEIGHT_POSITIONAL##__base( \
          const int64_t block_size,                                                                     \
          const int64_t output_size,                                                                    \
          const int64_t index_size,                                                                     \
          const int64_t data_size,                                                                      \
          const InType* input,                                                                         \
          const IndexType* indices,                                                                    \
          const int* lengths,                                                                          \
          const float* weights,                                                                        \
          const float* scale_bias,                                                                     \
          bool normalize_by_lengths,                                                                   \
          OutType* out) {                                                                              \
    EmbeddingLookupGenericSlow<                                                                        \
        IndexType,                                                                                     \
        InType,                                                                                        \
        OutType,                                                                                       \
        IS_WEIGHT_POSITIONAL>(                                                                         \
        block_size,                                                                                    \
        output_size,                                                                                   \
        index_size,                                                                                    \
        data_size,                                                                                     \
        input,                                                                                         \
        indices,                                                                                       \
        lengths,                                                                                       \
        weights,                                                                                       \
        scale_bias,                                                                                    \
        normalize_by_lengths,                                                                          \
        out);                                                                                          \
  }                                                                                                    \
  template <>                                                                                          \
  void EmbeddingLookup<IndexType, InType, OutType, IS_WEIGHT_POSITIONAL>(                              \
      const int64_t block_size,                                                                         \
      const int64_t output_size,                                                                        \
      const int64_t index_size,                                                                         \
      const int64_t data_size,                                                                          \
      const InType* input,                                                                             \
      const IndexType* indices,                                                                        \
      const int* lengths,                                                                              \
      const float* weights,                                                                            \
      const float* scale_bias,                                                                         \
      bool normalize_by_lengths,                                                                       \
      OutType* out) {                                                                                  \
    AVX2_FMA_DO(                                                                                       \
        EmbeddingLookup_##IndexTypeName##_##InTypeName##_##OutTypeName##_##IS_WEIGHT_POSITIONAL,       \
        block_size,                                                                                    \
        output_size,                                                                                   \
        index_size,                                                                                    \
        data_size,                                                                                     \
        input,                                                                                         \
        indices,                                                                                       \
        lengths,                                                                                       \
        weights,                                                                                       \
        scale_bias,                                                                                    \
        normalize_by_lengths,                                                                          \
        out);                                                                                          \
    BASE_DO(                                                                                           \
        EmbeddingLookup_##IndexTypeName##_##InTypeName##_##OutTypeName##_##IS_WEIGHT_POSITIONAL,       \
        block_size,                                                                                    \
        output_size,                                                                                   \
        index_size,                                                                                    \
        data_size,                                                                                     \
        input,                                                                                         \
        indices,                                                                                       \
        lengths,                                                                                       \
        weights,                                                                                       \
        scale_bias,                                                                                    \
        normalize_by_lengths,                                                                          \
        out);                                                                                          \
  }

EMBEDDING_SPECIALIZATION(int32_t, int32_t, float, float, float, float, false);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, float, float, float, float, false);
EMBEDDING_SPECIALIZATION(int32_t, int32_t, half, at::Half, float, float, false);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, half, at::Half, float, float, false);
EMBEDDING_SPECIALIZATION(int32_t, int32_t, uint8_t, uint8_t, float, float, false);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, uint8_t, uint8_t, float, float, false);

EMBEDDING_SPECIALIZATION(int32_t, int32_t, float, float, float, float, true);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, float, float, float, float, true);
EMBEDDING_SPECIALIZATION(int32_t, int32_t, half, at::Half, float, float, true);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, half, at::Half, float, float, true);
EMBEDDING_SPECIALIZATION(int32_t, int32_t, uint8_t, uint8_t, float, float, true);
EMBEDDING_SPECIALIZATION(int64_t, int64_t, uint8_t, uint8_t, float, float, true);

#undef EMBEDDING_SPECIALIZATION

} // namespace caffe2
