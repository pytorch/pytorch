#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include <math.h>

namespace at {
namespace native {

/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

static inline void upsample_1d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int nbatch,
    int nchannels,
    int input_width,
    int output_width) {
  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  if (input.defined()) {
    // Allow for empty batch size but not other dimensions
    bool valid_empty = false;
    valid_empty = input.size(0) == 0 && input.size(1) != 0 && input.size(2) != 0;
    
    TORCH_CHECK(
                (input.numel() != 0 || valid_empty) && input.dim() == 3,
                "Non-empty 3D data tensor expected but got a tensor with sizes ",
                input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 3, 0, nbatch);
    check_dim_size(grad_output, 3, 1, nchannels);
    check_dim_size(grad_output, 3, 2, output_width);
  }
}

static inline void upsample_2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int nbatch,
    int nchannels,
    int input_height,
    int input_width,
    int output_height,
    int output_width) {
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  if (input.defined()) {
    // Allow for empty batch size but not other dimensions
    bool valid_empty = false;
    valid_empty = input.size(0) == 0 && input.size(1) != 0 &&
      input.size(2) != 0 && input.size(3) != 0;
    TORCH_CHECK(
                (input.numel() != 0 || valid_empty) && input.dim() == 4,
                "Non-empty 4D data tensor expected but got a tensor with sizes ",
                input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, nchannels);
    check_dim_size(grad_output, 4, 2, output_height);
    check_dim_size(grad_output, 4, 3, output_width);
  }
}

static inline void upsample_3d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int nbatch,
    int nchannels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width) {
  TORCH_CHECK(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
          output_depth > 0 && output_height > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (D: ",
      input_depth,
      ", H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (D: ",
      output_depth,
      ", H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  if (input.defined()) {
    // Allow for empty batch size but not other dimensions
    bool valid_empty = false;
    valid_empty = input.size(0) == 0 && input.size(1) != 0 &&
      input.size(2) != 0 && input.size(3) != 0 && input.size(4) != 0;
    TORCH_CHECK(
                (input.numel() != 0 || valid_empty) && input.dim() == 5,
                "Non-empty 5D data tensor expected but got a tensor with sizes ",
                input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 5, 0, nbatch);
    check_dim_size(grad_output, 5, 1, nchannels);
    check_dim_size(grad_output, 5, 2, output_depth);
    check_dim_size(grad_output, 5, 3, output_height);
    check_dim_size(grad_output, 5, 4, output_width);
  }
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t compute_scales_value(
    const c10::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)(1.0 / scale.value())
                                                   : (accscalar_t)input_size / output_size;
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t compute_scales_value_backwards(
    const c10::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)scale.value()
                                                   : (accscalar_t)input_size / output_size;
}

template <typename accscalar_t>
__host__ __forceinline__ static accscalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners,
    const c10::optional<double> scale) {
  if (output_size > 1) {
    return align_corners ? (accscalar_t)(input_size - 1) / (output_size - 1)
                         :  compute_scales_value<accscalar_t>(scale, input_size, output_size);
  } else {
    return static_cast<accscalar_t>(0);
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

__device__ __forceinline__ static int nearest_neighbor_compute_source_index(
    const float scale,
    int dst_index,
    int input_size) {
  const int src_index =
      min(static_cast<int>(floorf(dst_index * scale)), input_size - 1);
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
  gpuAtomicAdd(
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
