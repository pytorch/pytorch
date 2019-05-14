#ifndef THCUNN_UPSAMPLING_H
#define THCUNN_UPSAMPLING_H

#include <THC/THCDeviceTensor.cuh>
#include <THC/THCAtomics.cuh>

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )


template<typename Acctype>
__host__ __forceinline__
static Acctype area_pixel_compute_scale(
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
static Acctype area_pixel_compute_source_index(
                          Acctype scale, int dst_index, bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    Acctype src_idx = scale * (dst_index + Acctype(0.5)) - Acctype(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < Acctype(0)) ? Acctype(0) : src_idx;
  }
}

__device__ __forceinline__
static int nearest_neighbor_compute_source_index(
                const float scale, int dst_index, int inputSize) {
  const int src_index = MIN(floor(dst_index * scale), inputSize - 1);
  return src_index;
}

template<typename Dtype>
__device__ __forceinline__
static Dtype upsampling_get_value_bounded(
  const THCDeviceTensor<Dtype, 4> data,
  int channel,
  int batch,
  int width,
  int height,
  int x,
  int y
) {
  int access_x = max(min(x, width - 1), 0);
  int access_y = max(min(y, height - 1), 0);
  return data[batch][channel][access_y][access_x];
}

template<typename Dtype, typename Acctype>
__device__ __forceinline__
static void upsampling_increment_value_bounded(
  const THCDeviceTensor<Dtype, 4> data,
  int channel,
  int batch,
  int width,
  int height,
  int x,
  int y,
  Acctype value
) {
  int access_x = max(min(x, width - 1), 0);
  int access_y = max(min(y, height - 1), 0);
  atomicAdd(
    data[batch][channel][access_y][access_x].data(),
    ScalarConvert<Acctype, Dtype>::to(value)
  );
}

// Based on https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template<typename Acctype>
__device__ __forceinline__
static Acctype cubic_convolution1(Acctype x, Acctype A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template<typename Acctype>
__device__ __forceinline__
static Acctype cubic_convolution2(Acctype x, Acctype A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template<typename Acctype>
__device__ __forceinline__
static void get_cubic_upsampling_coefficients(
  Acctype coeffs[4],
  Acctype t
) {
  Acctype A = -0.75;

  Acctype x1 = t;
  coeffs[0] = cubic_convolution2<Acctype>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<Acctype>(x1, A);

  // opposite coefficients
  Acctype x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<Acctype>(x2, A);
  coeffs[3] = cubic_convolution2<Acctype>(x2 + 1.0, A);
}

template<typename Dtype, typename Acctype>
__device__ __forceinline__
static Acctype cubic_interp1d(
  Dtype x0,
  Dtype x1,
  Dtype x2,
  Dtype x3,
  Acctype t
) {
  Acctype coeffs[4];
  get_cubic_upsampling_coefficients<Acctype>(coeffs, t);

  return x0 * coeffs[0]
    + x1 * coeffs[1]
    + x2 * coeffs[2]
    + x3 * coeffs[3];
}

#endif
