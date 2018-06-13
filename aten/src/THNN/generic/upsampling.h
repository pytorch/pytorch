#ifndef THNN_UPSAMPLING_H
#define THNN_UPSAMPLING_H

template<typename T>
static inline T upsampling_compute_scale(
                          int inputSize, int outputSize, bool align_corners) {
  if (outputSize > 1) {
    return align_corners ? (T) (inputSize - 1) / (outputSize - 1)
                         : (T) inputSize / outputSize;
  } else {
    return T(0);
  }
}

template<typename T>
static inline T upsampling_compute_source_index(
                          T scale, int dst_index, bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    T src_idx = scale * (dst_index + 0.5) - 0.5;
    return src_idx < 0 ? T(0) : src_idx;
  }
}


#endif


