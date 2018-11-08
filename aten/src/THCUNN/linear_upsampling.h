#ifndef THCUNN_LINEAR_UPSAMPLING_H
#define THCUNN_LINEAR_UPSAMPLING_H

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )


template<typename Acctype>
__host__ __forceinline__
static Acctype linear_upsampling_compute_scale(
                          int inputSize, int outputSize, bool align_corners) {
  if (outputSize > 1) {
    return align_corners ? (Acctype) (inputSize - 1) / (outputSize - 1)
                         : (Acctype) inputSize / outputSize;
  } else {
    return Acctype(0);
  }
}

template<typename Acctype>
__device__ __forceinline__
static Acctype linear_upsampling_compute_source_index(
                          Acctype scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    Acctype src_idx = scale * (dst_index + Acctype(0.5)) - Acctype(0.5);
    return src_idx < Acctype(0) ? Acctype(0) : src_idx;
  }
}

__device__ __forceinline__
static int nearest_neighbor_compute_source_index(
		const float scale, int dst_index, int inputSize) {
  const int src_index = MIN(floor(dst_index * scale), inputSize - 1);
  return src_index;
}
#endif

