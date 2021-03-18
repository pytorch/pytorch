#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace meta {

TORCH_META_FUNC(upsample_bicubic2d) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output(full_output_size, input.options());
}

TORCH_META_FUNC(upsample_bicubic2d_backward) (
  const Tensor& grad_output,
  IntArrayRef output_size,
  IntArrayRef input_size,
  bool align_corners,
  c10::optional<double> scales_h,
  c10::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ", grad_output.dim());

  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output(input_size, grad_output.options());
}

} // namespace meta
namespace native {
namespace {

template <typename scalar_t>
static void upsample_bicubic2d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int64_t output_y = 0; output_y < output_height; output_y++) {
      for (int64_t output_x = 0; output_x < output_width; output_x++) {
        const scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];

        for (int64_t c = 0; c < channels * nbatch; ++c) {
          out[0] = in[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    return;
  }

  // Bicubic interpolation
  const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
      input_height, output_height, align_corners, scales_h);
  const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
      input_width, output_width, align_corners, scales_w);

  for (int64_t output_y = 0; output_y < output_height; output_y++) {
    for (int64_t output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      const scalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
      int64_t input_x = floorf(real_x);
      const scalar_t t_x = real_x - input_x;

      const scalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
      int64_t input_y = floorf(real_y);
      const scalar_t t_y = real_y - input_y;

      for (int64_t c = 0; c < channels * nbatch; c++) {
        scalar_t coefficients[4];

        // Interpolate 4 times in the x direction
        for (int64_t i = 0; i < 4; i++) {
          coefficients[i] = cubic_interp1d<scalar_t>(
              upsample_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x - 1, input_y - 1 + i),
              upsample_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 0, input_y - 1 + i),
              upsample_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 1, input_y - 1 + i),
              upsample_get_value_bounded<scalar_t>(
                  in, input_width, input_height, input_x + 2, input_y - 1 + i),
              t_x);
        }

        // Interpolate in the y direction using x interpolations
        out[output_y * output_width + output_x] = cubic_interp1d<scalar_t>(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            t_y);

        // Move to next channel
        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }
}

template <typename scalar_t>
static void upsample_bicubic2d_backward_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  channels = channels * nbatch;

  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    for (int64_t output_y = 0; output_y < output_height; output_y++) {
      for (int64_t output_x = 0; output_x < output_width; output_x++) {
        scalar_t* in = &idata[output_y * input_width + output_x];
        scalar_t* out = &odata[output_y * output_width + output_x];
        for (int64_t c = 0; c < channels; ++c) {
          in[0] = out[0];
          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    return;
  }

  const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
      input_height, output_height, align_corners, scales_h);
  const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
      input_width, output_width, align_corners, scales_w);

  for (int64_t output_y = 0; output_y < output_height; output_y++) {
    for (int64_t output_x = 0; output_x < output_width; output_x++) {
      scalar_t* in = idata;
      scalar_t* out = odata;

      const scalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
      int64_t input_x = floorf(real_x);
      scalar_t t_x = real_x - input_x;

      const scalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
      int64_t input_y = floorf(real_y);
      scalar_t t_y = real_y - input_y;

      scalar_t x_coeffs[4];
      scalar_t y_coeffs[4];

      get_cubic_upsample_coefficients<scalar_t>(x_coeffs, t_x);
      get_cubic_upsample_coefficients<scalar_t>(y_coeffs, t_y);

      for (int64_t c = 0; c < channels; c++) {
        scalar_t out_value = out[output_y * output_width + output_x];

        for (int64_t i = 0; i < 4; i++) {
          for (int64_t j = 0; j < 4; j++) {
            upsample_increment_value_bounded<scalar_t>(
                in,
                input_width,
                input_height,
                input_x - 1 + i,
                input_y - 1 + j,
                out_value * y_coeffs[j] * x_coeffs[i]);
          }
        }

        in += input_width * input_height;
        out += output_width * output_height;
      }
    }
  }
}

static void upsample_bicubic2d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  auto input = input_.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bicubic2d", [&] {
    auto* idata = input.data_ptr<scalar_t>();
    auto* odata = output.data_ptr<scalar_t>();

    upsample_bicubic2d_out_frame<scalar_t>(
        odata,
        idata,
        input_height,
        input_width,
        output_height,
        output_width,
        nbatch,
        channels,
        align_corners,
        scales_h,
        scales_w);
  });
}

static void upsample_bicubic2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  auto grad_output = grad_output_.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_bicubic2d_backward", [&] {
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();

        upsample_bicubic2d_backward_out_frame<scalar_t>(
            odata,
            idata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            align_corners,
            scales_h,
            scales_w);
      });
}
} // namespace

TORCH_IMPL_FUNC(upsample_bicubic2d_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output
) {
  output.zero_();
  upsample_bicubic2d_kernel(output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input
) {
  grad_input.zero_();
  upsample_bicubic2d_backward_kernel(grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

// vec variants

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_bicubic2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_bicubic2d(input, osize, align_corners, scale_h, scale_w);
}

Tensor upsample_bicubic2d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_bicubic2d_backward(grad_output, osize, input_size, align_corners, scale_h, scale_w);
}

} // namespace native
} // namespace at
