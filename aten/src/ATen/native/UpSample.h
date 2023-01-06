#pragma once

#include <math.h>

#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

/**
 * Note [compute_scales_value]
 * Note [area_pixel_compute_scale]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Interpolate with scale_factor can have different behaviors
 * depending on the value of recompute_scale_factor:
 *
 * - With recompute_scale_factor = True (current default behavior):
 * the scale_factor, when provided by the user, are used to calculate
 * the output size. The input size and the computed output_size
 * are then used to infer new values for the scales which are
 * used in the interpolation.  Because floating-point math is not exact,
 * this may be a different value from the user-supplied scales.
 *
 * - With recompute_scale_factor = False (which will be the default
 * behavior starting 1.5.0):
 * the behavior follows opencv logic, and the scales provided by
 * the user are the ones used in the interpolation calculations.
 *
 * If the scales are not provided or if they are provided but
 * recompute_scale_factor is set to True (default behavior), the scales
 * are computed from the input and the output size;
 *
 *
 * When the scales are inferred from the input and output sizes,
 * we view each pixel as an area, idx + 0.5 as its center index.
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

namespace at {
namespace native {

namespace upsample {

TORCH_API c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,  // Full input tensor size.
    at::OptionalIntArrayRef output_size,
    c10::optional<c10::ArrayRef<double>> scale_factors);

inline c10::optional<double> get_scale_value(c10::optional<c10::ArrayRef<double>> scales, int idx) {
  if (!scales) {
    return c10::nullopt;
  }
  return scales->at(idx);
}

} // namespace upsample

using scale_t = c10::optional<double>;
using upsampling_nearest1d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_w);
using _upsampling_nearest_exact1d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_w);
using upsampling_nearest2d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_h, scale_t scales_w);
using _upsampling_nearest_exact2d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_h, scale_t scales_w);
using upsampling_nearest3d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_d, scale_t scales_h, scale_t scales_w);
using _upsampling_nearest_exact3d = void(*)(const Tensor& output, const Tensor& input, scale_t scales_d, scale_t scales_h, scale_t scales_w);
using upsampling_linear1d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_w);
using upsampling_bilinear2d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
using _upsampling_bilinear2d_aa = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
using upsampling_trilinear3d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_d, scale_t scales_h, scale_t scales_w);
using upsampling_bicubic2d = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
using _upsampling_bicubic2d_aa = void(*)(const Tensor& output, const Tensor& input, bool align_corners, scale_t scales_h, scale_t scales_w);
DECLARE_DISPATCH(upsampling_nearest1d, upsample_nearest1d_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact1d, _upsample_nearest_exact1d_kernel);
DECLARE_DISPATCH(upsampling_nearest2d, upsample_nearest2d_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact2d, _upsample_nearest_exact2d_kernel);
DECLARE_DISPATCH(upsampling_nearest3d, upsample_nearest3d_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact3d, _upsample_nearest_exact3d_kernel);
DECLARE_DISPATCH(upsampling_nearest1d, upsample_nearest1d_backward_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact1d, _upsample_nearest_exact1d_backward_kernel);
DECLARE_DISPATCH(upsampling_nearest2d, upsample_nearest2d_backward_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact2d, _upsample_nearest_exact2d_backward_kernel);
DECLARE_DISPATCH(upsampling_nearest3d, upsample_nearest3d_backward_kernel);
DECLARE_DISPATCH(_upsampling_nearest_exact3d, _upsample_nearest_exact3d_backward_kernel);
DECLARE_DISPATCH(upsampling_linear1d, upsample_linear1d_kernel);
DECLARE_DISPATCH(upsampling_bilinear2d, upsample_bilinear2d_kernel);
DECLARE_DISPATCH(_upsampling_bilinear2d_aa, _upsample_bilinear2d_aa_kernel);
DECLARE_DISPATCH(upsampling_trilinear3d, upsample_trilinear3d_kernel);
DECLARE_DISPATCH(upsampling_linear1d, upsample_linear1d_backward_kernel);
DECLARE_DISPATCH(upsampling_bilinear2d, upsample_bilinear2d_backward_kernel);
DECLARE_DISPATCH(_upsampling_bilinear2d_aa, _upsample_bilinear2d_aa_backward_kernel);
DECLARE_DISPATCH(upsampling_trilinear3d, upsample_trilinear3d_backward_kernel);
DECLARE_DISPATCH(upsampling_bicubic2d, upsample_bicubic2d_kernel);
DECLARE_DISPATCH(_upsampling_bicubic2d_aa, _upsample_bicubic2d_aa_kernel);
DECLARE_DISPATCH(_upsampling_bicubic2d_aa, _upsample_bicubic2d_aa_backward_kernel);

static C10_UNUSED std::array<int64_t, 3> upsample_1d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  return {nbatch, channels, output_width};
}

static C10_UNUSED std::array<int64_t, 4> upsample_2d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

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

  return {nbatch, channels, output_height, output_width};
}

static C10_UNUSED
std::array<int64_t, 5> upsample_3d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

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


  return {nbatch, channels, output_depth, output_height, output_width};
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
    // Allow for empty batch size but not other dimensions
    TORCH_CHECK(
                (input.numel() != 0 ||
                 (input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0)
                 ) &&
                input.dim() == 4,
                "Non-empty 4D data tensor expected but got a tensor with sizes ",
                input.sizes());
  } else if (grad_output.defined()) {
    check_dim_size(grad_output, 4, 0, nbatch);
    check_dim_size(grad_output, 4, 1, nchannels);
    check_dim_size(grad_output, 4, 2, output_height);
    check_dim_size(grad_output, 4, 3, output_width);
  }
}

template <typename scalar_t>
static inline scalar_t compute_scales_value(
    const c10::optional<double> scale,
    int64_t input_size,
    int64_t output_size) {
      // see Note [compute_scales_value]
      // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
      return (scale.has_value() && scale.value() > 0.)
          ? static_cast<scalar_t>(1.0 / scale.value())
          : (static_cast<scalar_t>(input_size) / output_size);
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    const c10::optional<double> scale) {
  // see Note [area_pixel_compute_scale]
  if(align_corners) {
    if(output_size > 1) {
      return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<scalar_t>(0);
    }
  } else {
    return compute_scales_value<scalar_t>(scale, input_size, output_size);
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
    scalar_t src_idx = scale * (dst_index + static_cast<scalar_t>(0.5)) -
        static_cast<scalar_t>(0.5);
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
    return (!cubic && src_idx < static_cast<scalar_t>(0)) ? scalar_t(0)
                                                          : src_idx;
  }
}

static inline int64_t nearest_neighbor_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
  // Index computation matching OpenCV INTER_NEAREST
  // which is buggy and kept for BC
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

static inline int64_t nearest_neighbor_exact_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
  // index_f32 = (output_index + 0.5) * scale - 0.5
  // input_index = round(index_f32)
  // Same as Pillow and Scikit-Image/Scipy ndi.zoom
  const int64_t src_index =
      std::min(static_cast<int64_t>(floorf((dst_index + 0.5) * scale)), input_size - 1);
  return src_index;
}

static inline int64_t nearest_idx(
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    c10::optional<double> scales) {
  // This method specificly treats cases: output_size == input_size or
  // output_size == 2 * input_size, that we would like to get rid of
  // We keep this method for BC and consider as deprecated.
  // See nearest_exact_idx as replacement
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    return output_index;
  } else if (output_size == 2 * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    float scale = compute_scales_value<float>(scales, input_size, output_size);
    return nearest_neighbor_compute_source_index(scale, output_index, input_size);
  }
}

static inline int64_t nearest_exact_idx(
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    c10::optional<double> scales) {
  float scale = compute_scales_value<float>(scales, input_size, output_size);
    return nearest_neighbor_exact_compute_source_index(scale, output_index, input_size);
}

// Define a typedef to dispatch to nearest_idx or nearest_exact_idx
typedef int64_t (*nearest_idx_fn_t)(int64_t, int64_t, int64_t, c10::optional<double>);

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

template<typename scalar_t>
static inline void compute_source_index_and_lambda(
    int64_t& input_index0,
    int64_t& input_index1,
    scalar_t& lambda0,
    scalar_t& lambda1,
    scalar_t ratio,
    int64_t output_index,
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {
  if (output_size == input_size) {
    // scale_factor = 1, simply copy
    input_index0 = output_index;
    input_index1 = output_index;
    lambda0 = static_cast<scalar_t>(1);
    lambda1 = static_cast<scalar_t>(0);
  } else {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto real_input_index =
        area_pixel_compute_source_index<opmath_t>(
            ratio, output_index, align_corners, /*cubic=*/false);
    // when `real_input_index` becomes larger than the range the floating point
    // type can accurately represent, the type casting to `int64_t` might exceed
    // `input_size - 1`, causing overflow. So we guard it with `std::min` below.
    input_index0 = std::min(static_cast<int64_t>(real_input_index), input_size - 1);
    int64_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;
    lambda1 = std::min(
      std::max(real_input_index - input_index0, static_cast<opmath_t>(0)),
      static_cast<opmath_t>(1)
    );
    lambda0 = static_cast<scalar_t>(1.) - lambda1;
  }
}

} // namespace native
} // namespace at
