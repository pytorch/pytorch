#ifndef THNN_UPSAMPLING_H
#define THNN_UPSAMPLING_H

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

template<typename T>
static T upsampling_get_value_bounded(T* data, int width, int height, int x, int y) {
  int access_x = std::max(std::min(x, width - 1), 0);
  int access_y = std::max(std::min(y, height - 1), 0);
  return data[access_y * width + access_x];
}

template<typename T>
static void upsampling_increment_value_bounded(
  T* data,
  int width,
  int height,
  int x,
  int y,
  T value
) {
  int access_x = std::max(std::min(x, width - 1), 0);
  int access_y = std::max(std::min(y, height - 1), 0);
  data[access_y * width + access_x] += value;
}

// Based on https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename T>
static inline T cubic_convolution1(T x, T A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template<typename T>
static inline T cubic_convolution2(T x, T A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template<typename T>
static inline void get_cubic_upsampling_coefficients(T coeffs[4], T t) {
  T A = -0.75;

  T x1 = t;
  coeffs[0] = cubic_convolution2<T>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<T>(x1, A);

  // opposite coefficients
  T x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<T>(x2, A);
  coeffs[3] = cubic_convolution2<T>(x2 + 1.0, A);
}

template<typename T>
static inline T cubic_interp1d(
  T x0,
  T x1,
  T x2,
  T x3,
  T t
) {
  T coeffs[4];
  get_cubic_upsampling_coefficients<T>(coeffs, t);

  return x0 * coeffs[0]
    + x1 * coeffs[1]
    + x2 * coeffs[2]
    + x3 * coeffs[3];
}

#endif
