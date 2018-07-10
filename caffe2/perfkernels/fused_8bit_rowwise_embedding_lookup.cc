#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"

#include "caffe2/core/types.h"
#include "caffe2/perfkernels/common.h"
#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Base implementation does runtime dispatch for each segment of reduction
template <
    typename IndexType,
    typename InType,
    typename OutType,
    bool IS_WEIGHT_POSITIONAL = false>
static void Fused8BitRowwiseEmbeddingLookupGenericSlow(
    const TIndex block_size,
    const TIndex output_size,
    const TIndex index_size,
    const TIndex data_size,
    const InType* input,
    const IndexType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for sum reducer
    bool normalize_by_lengths,
    OutType* out) {
  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 8 / sizeof(InType);
  const TIndex fused_block_size = block_size + scale_bias_offset;
  TIndex current = 0;
  for (int m = 0; m < output_size; ++m) {
    memset(out, 0, sizeof(OutType) * block_size);
    EigenVectorArrayMap<OutType> out_vector(out, block_size);
    for (int i = 0; i < lengths[m]; ++i) {
      CAFFE_ENFORCE_LT(current, index_size);
      TIndex idx = indices[current];
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

      TypedAxpy<InType, OutType>(
          block_size, scale, input + fused_block_size * indices[current], out);

      out_vector += bias;

      ++current;
    }
    if (normalize_by_lengths && lengths[m]) {
      // hack: context is not really used
      math::Scale<OutType, CPUContext>(
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
#define FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(                                    \
    IndexType, InType, OutType)                                                         \
  void                                                                                  \
      Fused8BitRowwiseEmbeddingLookup_##IndexType##_##InType##_##OutType##_false__base( \
          const TIndex block_size,                                                      \
          const TIndex output_size,                                                     \
          const TIndex index_size,                                                      \
          const TIndex data_size,                                                       \
          const InType* input,                                                          \
          const IndexType* indices,                                                     \
          const int* lengths,                                                           \
          const float* weights,                                                         \
          bool normalize_by_lengths,                                                    \
          OutType* out) {                                                               \
    Fused8BitRowwiseEmbeddingLookupGenericSlow<                                         \
        IndexType,                                                                      \
        InType,                                                                         \
        OutType,                                                                        \
        false>(                                                                         \
        block_size,                                                                     \
        output_size,                                                                    \
        index_size,                                                                     \
        data_size,                                                                      \
        input,                                                                          \
        indices,                                                                        \
        lengths,                                                                        \
        weights,                                                                        \
        normalize_by_lengths,                                                           \
        out);                                                                           \
  }                                                                                     \
  template <>                                                                           \
  void Fused8BitRowwiseEmbeddingLookup<IndexType, InType, OutType, false>(              \
      const TIndex block_size,                                                          \
      const TIndex output_size,                                                         \
      const TIndex index_size,                                                          \
      const TIndex data_size,                                                           \
      const InType* input,                                                              \
      const IndexType* indices,                                                         \
      const int* lengths,                                                               \
      const float* weights,                                                             \
      bool normalize_by_lengths,                                                        \
      OutType* out) {                                                                   \
    const int32_t one = 1;                                                              \
    CAFFE_ENFORCE_EQ(                                                                   \
        reinterpret_cast<const uint8_t*>(&one)[0],                                      \
        1,                                                                              \
        "Fused8BitRowwiseEmbeddingLookup is not supported on this platform");           \
    AVX2_FMA_DO(                                                                        \
        Fused8BitRowwiseEmbeddingLookup_##IndexType##_##InType##_##OutType##_false,     \
        block_size,                                                                     \
        output_size,                                                                    \
        index_size,                                                                     \
        data_size,                                                                      \
        input,                                                                          \
        indices,                                                                        \
        lengths,                                                                        \
        weights,                                                                        \
        normalize_by_lengths,                                                           \
        out);                                                                           \
    BASE_DO(                                                                            \
        Fused8BitRowwiseEmbeddingLookup_##IndexType##_##InType##_##OutType##_false,     \
        block_size,                                                                     \
        output_size,                                                                    \
        index_size,                                                                     \
        data_size,                                                                      \
        input,                                                                          \
        indices,                                                                        \
        lengths,                                                                        \
        weights,                                                                        \
        normalize_by_lengths,                                                           \
        out);                                                                           \
  }

FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(int32_t, uint8_t, float);
FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION(int64_t, uint8_t, float);

#undef FUSED_8BIT_ROWWISE_EMBEDDING_SPECIALIZATION

} // namespace caffe2
