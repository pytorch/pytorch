#include "caffe2/core/types.h"

namespace caffe2 {

// TODO(msmelyan): implement code generator for implementation based on
// following parameters:
//   index type: int32, int64 (encoded in function name)
//   embedding data type: float16, float32 (encoded in function name)
//   output type: float (encoded in function name)
//   weighted reduction: whether `weights` is nullptr or not
//   normalization (divide by lengths[i]): whether normalize_by_lengths is true
//   block size: 32, 64, 128, generic

// For now just invoke base implementation (this entire file can be autogenned)
#define EMBEDDING_SPECIALIZATION(IndexType, InType, OutType)                 \
  void EmbeddingLookup_##IndexType##_##InType##_##OutType##__avx2_fma(       \
      const TIndex block_size,                                               \
      const TIndex output_size,                                              \
      const TIndex index_size,                                               \
      const TIndex data_size,                                                \
      const InType* input,                                                   \
      const IndexType* indices,                                              \
      const int* lengths,                                                    \
      const float* weights,                                                  \
      bool normalize_by_lengths,                                             \
      OutType* out) {                                                        \
    decltype(EmbeddingLookup_##IndexType##_##InType##_##OutType##__avx2_fma) \
        EmbeddingLookup_##IndexType##_##InType##_##OutType##__base;          \
    EmbeddingLookup_##IndexType##_##InType##_##OutType##__base(              \
        block_size,                                                          \
        output_size,                                                         \
        index_size,                                                          \
        data_size,                                                           \
        input,                                                               \
        indices,                                                             \
        lengths,                                                             \
        weights,                                                             \
        normalize_by_lengths,                                                \
        out);                                                                \
  }

EMBEDDING_SPECIALIZATION(int32_t, float, float);
EMBEDDING_SPECIALIZATION(int64_t, float, float);
EMBEDDING_SPECIALIZATION(int32_t, float16, float);
EMBEDDING_SPECIALIZATION(int64_t, float16, float);

#undef EMBEDDING_SPECIALIZATION

} // namespace caffe2
