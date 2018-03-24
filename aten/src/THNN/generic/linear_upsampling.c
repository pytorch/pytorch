#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/linear_upsampling.c"
#else

static inline float THNN_(linear_upsampling_compute_scale)(
                          int inputSize, int outputSize, bool align_corners) {
  if (outputSize > 1) {
    return align_corners ? (float) (inputSize - 1) / (outputSize - 1)
                         : (float) inputSize / outputSize;
  } else {
    return 0.f;
  }
}

static inline float THNN_(linear_upsampling_compute_source_index)(
                          float scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    float src_idx = scale * (dst_index + 0.5) - 0.5;
    return src_idx < 0 ? 0.f : src_idx;
  }
}


#endif

