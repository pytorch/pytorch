#include <math.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace native {

static inline void upsample_1d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t nbatch,
    int64_t nchannels,
    int64_t input_width,
    int64_t output_width) {
  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  if (input.defined()) {
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 3,
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
    int64_t nbatch,
    int64_t nchannels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
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
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 4,
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
    int64_t nbatch,
    int64_t nchannels,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
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
    TORCH_CHECK(
        input.numel() != 0 && input.dim() == 5,
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

template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  /* We view each pixel as an area, idx + 0.5 as its center index.
   * Here is an example formula in 1D case.
   * if align_corners: center of two corner pixel areas are preserved,
   *     (0.5, 0.5) -> (0.5, 0.5),
   *     (input_size - 0.5, 0.5) -> (output_size - 0.5)
   *     scale = (input_size - 0.5 - 0.5) / (output_size - 0.5 - 0.5)
   *     src_index + 0.5 - 0.5 = scale * (dst_index + 0.5 - 0.5)
   * if not align_corners: the whole range is scaled accordingly
   *     scale = input_size / output_size
   *     src_idx + 0.5 = scale * (dst_index + 0.5)
   */
  if (output_size > 1) {
    return align_corners
        ? static_cast<scalar_t>(input_size - 1) / (output_size - 1)
        : static_cast<scalar_t>(input_size) / output_size;
  } else {
    return scalar_t(0);
  }
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,
    int64_t dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
    // [Note] Follow Opencv resize logic:
    // We allow negative src_idx here and later will use
    //   dx = src_idx - floorf(src_idx)
    // to compute the "distance"(which affects weights).
    // For linear modes, weight distribution doesn't matter
    // for negative indices as they use 2 pixels to interpolate.
    // For example, [-1, 0], they both use pixel 0 value so it
    // doesn't affect if we bound the src_idx to 0 or not.
    // TODO: Our current linear mode impls use unbound indices
    // where we should and then remove this cubic flag.
    // This matters in cubic mode, as we might need [-1, 0, 1, 2]
    // to interpolate and the weights can be affected.
    return (!cubic && src_idx < 0) ? scalar_t(0) : src_idx;
  }
}

static inline int64_t nearest_neighbor_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

template <typename scalar_t>
static scalar_t upsample_get_value_bounded(
    scalar_t* data,
    int64_t width,
    int64_t height,
    int64_t x,
    int64_t y) {
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));
  return data[access_y * width + access_x];
}

template <typename scalar_t>
static void upsample_increment_value_bounded(
    scalar_t* data,
    int64_t width,
    int64_t height,
    int64_t x,
    int64_t y,
    scalar_t value) {
  int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));
  int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));
  data[access_y * width + access_x] += value;
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename scalar_t>
static inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
static inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
static inline void get_cubic_upsample_coefficients(
    scalar_t coeffs[4],
    scalar_t t) {
  scalar_t A = -0.75;

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  // opposite coefficients
  scalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}

template <typename scalar_t>
static inline scalar_t cubic_interp1d(
    scalar_t x0,
    scalar_t x1,
    scalar_t x2,
    scalar_t x3,
    scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsample_coefficients<scalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

} // namespace native
} // namespace at
