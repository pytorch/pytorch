#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/Atomic.cuh>

#include <c10/util/Optional.h>

#include <math.h>

namespace at {
namespace native {

namespace upsample {
// TODO: Remove duplicate declaration.
TORCH_API c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,  // Full input tensor size.
    c10::optional<c10::IntArrayRef> output_size,
    c10::optional<c10::ArrayRef<double>> scale_factors);
} // namespace upsample

namespace upsample_cuda {

// TODO: Remove duplication with Upsample.h (CPU).
inline c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx) {
  if (!scales) {
    return nullopt;
  }
  return scales->at(idx);
}

} // namespace upsample_cuda


/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

// NOTE [ Nearest neighbor upsampling kernel implementation ]
//
// The nearest neighbor upsampling kernel implementation is symmetrical as
// expected. We launch kernels with threads mapping to destination tensors where
// kernels write data to, each thread reads data from the source tensor, this
// means:
// 1. In the forward kernel,
//      src_xxx refers to properties of input tensors;
//      dst_xxx refers to properties of output tensors;
//      scale_factor is the ratio of src_size to dst_size;
// 2. In the backward kernel,
//      src_xxx refers to properties of grad_output tensors;
//      dst_xxx refers to properties of grad_input tensors;
//      scale_factor is the ratio of src_size to dst_size;
//
// Because of this, we need to take the reciprocal of the scale defined by
// upsample layer during forward path. The motivation is to avoid slow
// division in the kernel code, so we can use faster multiplication instead.
// This is not necessary during backward path, since the scale_factor is already
// the reciprocal of corresponding scale_factor used in the forward path due to
// the swap of source and destination tensor.
//
// Similarly, since the mapping from grad_input to grad_output during backward
// is the reverse of the mapping of output to input, we need to have opposite
// mapping functions to compute the source index.

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t compute_scales_value(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)(1.0 / scale.value())
                                                   : (accscalar_t)src_size / dst_size;
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t compute_scales_value_backwards(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)scale.value()
                                                   : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners,
    const c10::optional<double> scale) {
  if(align_corners) {
    if(output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    }
    else {
      return static_cast<accscalar_t>(0);
    }
  }
  else{
    return compute_scales_value<accscalar_t>(scale, input_size, output_size);
  }
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
__device__ __forceinline__ static int nearest_neighbor_compute_source_index(
    const float scale,
    int dst_index,
    int input_size) {
  // index_f32 = (output_index) * scale
  // input_index = round(index_f32)
  // Same as a buggy OpenCV INTER_NEAREST
  // We keep this method for BC and consider as deprecated.
  // See nearest_neighbor_exact_compute_source_index as replacement
  const int src_index =
      min(static_cast<int>(floorf((dst_index) * scale)), input_size - 1);
  return src_index;
}

__device__ __forceinline__ static int nearest_neighbor_exact_compute_source_index(
    const float scale,
    int dst_index,
    int input_size) {
  // index_f32 = (output_index + 0.5) * scale - 0.5
  // input_index = round(index_f32)
  // Same as Pillow and Scikit-Image/Scipy ndi.zoom
  const int src_index =
      min(static_cast<int>(floorf((dst_index + 0.5) * scale)), input_size - 1);
  return src_index;
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
__device__ __forceinline__ static int nearest_neighbor_bw_compute_source_index(
    const float scale,
    int dst_index,
    int output_size) {
  // Equivalent to buggy OpenCV INTER_NEAREST
  // We keep this method for BC and consider as deprecated.
  // See nearest_neighbor_exact_bw_compute_source_index as replacement
  const int src_index =
      min(static_cast<int>(ceilf(dst_index * scale)), output_size);
  return src_index;
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
__device__ __forceinline__ static int nearest_neighbor_exact_bw_compute_source_index(
    const float scale,
    int dst_index,
    int output_size) {
  // Equivalent to Pillow and Scikit-Image/Scipy ndi.zoom
  const int src_index =
      min(static_cast<int>(ceilf(dst_index * scale - 0.5)), output_size);
  return src_index;
}

/* Used by UpSampleBicubic2d.cu */
template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(
    const PackedTensorAccessor64<scalar_t, 4>& data,
    int batch,
    int channel,
    int height,
    int width,
    int y,
    int x) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  return data[batch][channel][access_y][access_x];
}

/* Used by UpSampleBicubic2d.cu */
template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static void upsample_increment_value_bounded(
    PackedTensorAccessor64<scalar_t, 4>& data,
    int batch,
    int channel,
    int height,
    int width,
    int y,
    int x,
    accscalar_t value) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  /* TODO: result here is truncated to scalar_t,
     check: https://github.com/pytorch/pytorch/pull/19630#discussion_r281426912
   */
  gpuAtomicAddNoReturn(
      &data[batch][channel][access_y][access_x], static_cast<scalar_t>(value));
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution1(
    accscalar_t x,
    accscalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_convolution2(
    accscalar_t x,
    accscalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename accscalar_t>
__device__ __forceinline__ static void get_cubic_upsampling_coefficients(
    accscalar_t coeffs[4],
    accscalar_t t) {
  accscalar_t A = -0.75;

  accscalar_t x1 = t;
  coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

  // opposite coefficients
  accscalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t cubic_interp1d(
    scalar_t x0,
    scalar_t x1,
    scalar_t x2,
    scalar_t x3,
    accscalar_t t) {
  accscalar_t coeffs[4];
  get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

} // namespace native
} // namespace at
