#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
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

        for (int64_t c = 0; c < channels; ++c) {
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

static void upsample_bicubic2d_out_cpu_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  upsample_2d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_height, output_width});
  output.zero_();

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

static void upsample_bicubic2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
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

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

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

Tensor& upsample_bicubic2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_bicubic2d_out_cpu_template(
      output, input, output_size, align_corners, scales_h, scales_w);
  return output;
}

Tensor upsample_bicubic2d_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  upsample_bicubic2d_out_cpu_template(
      output, input, output_size, align_corners, scales_h, scales_w);
  return output;
}

Tensor& upsample_bicubic2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_bicubic2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  return grad_input;
}

Tensor upsample_bicubic2d_backward_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_bicubic2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  return grad_input;
}

} // namespace native
} // namespace at
