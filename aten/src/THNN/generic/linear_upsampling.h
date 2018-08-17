#ifndef THNN_LINEAR_UPSAMPLING_H
#define THNN_LINEAR_UPSAMPLING_H

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )


template<typename T>
static inline T linear_upsampling_compute_scale(
                          int inputSize, int outputSize, bool align_corners) {
  /* We view each pixel as an area, idx + 0.5 as its center index.
   * Here is an example formula in 1D case.
   * if align_corners: center of two corner pixel areas are preserved,
   *     (0.5, 0.5) -> (0.5, 0.5),
   *     (inputSize - 0.5, 0.5) -> (outputSize - 0.5)
   *     scale = (inputSize - 0.5 - 0.5) / (outputSize - 0.5 - 0.5)
   *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
   * if not align_corners: the whole range is scaled accordingly
   *     scale = inputSize / outputSize
   *     src_idx + 0.5 = scale * (dst_index + 0.5)
   */
  if (outputSize > 1) {
    return align_corners ? (T) (inputSize - 1) / (outputSize - 1)
                         : (T) inputSize / outputSize;
  } else {
    return T(0);
  }
}

template<typename T>
static inline T linear_upsampling_compute_source_index(
                          T scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    T src_idx = scale * (dst_index + 0.5) - 0.5;
    return src_idx < 0 ? T(0) : src_idx;
  }
}

static inline int nearest_neighbor_compute_source_index(
		const float scale, int dst_index, int inputSize) {
  const int src_index = MIN(floorf(dst_index * scale), inputSize - 1);
  return src_index;
}


#endif

