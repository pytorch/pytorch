// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

namespace at {
namespace native {
namespace {

template <typename scalar_t>
static void upsample_bilinear2d_out_frame(
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

  // special case: just copy
  if (input_height == output_height && input_width == output_width) {
    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 = h2;

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 = w2;
        const scalar_t* pos1 = &idata[h1 * input_width + w1];
        scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_width * input_height;
          pos2 += output_width * output_height;
        }
      }
    }
    return;
  }
  const scalar_t rheight = area_pixel_compute_scale<scalar_t>(
      input_height, output_height, align_corners, scales_h);

  const scalar_t rwidth = area_pixel_compute_scale<scalar_t>(
       input_width, output_width, align_corners, scales_w);

  for (int64_t h2 = 0; h2 < output_height; ++h2) {
    const scalar_t h1r = area_pixel_compute_source_index<scalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);

    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;

    const scalar_t h1lambda = h1r - h1;
    const scalar_t h0lambda = static_cast<scalar_t>(1.) - h1lambda;

    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const scalar_t w1r = area_pixel_compute_source_index<scalar_t>(
          rwidth, w2, align_corners, /*cubic=*/false);

      const int64_t w1 = w1r;
      const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;

      const scalar_t w1lambda = w1r - w1;
      const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;
      const scalar_t* pos1 = &idata[h1 * input_width + w1];
      scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
            h1lambda *
                (w0lambda * pos1[h1p * input_width] +
                 w1lambda * pos1[h1p * input_width + w1p]);
        pos1 += input_width * input_height;
        pos2 += output_width * output_height;
      }
    }
  }
}

template <typename scalar_t>
static void upsample_bilinear2d_backward_out_frame(
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

  // special case: same-size matching grids
  if (input_height == output_height && input_width == output_width) {
    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 = h2;
      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 = w2;
        scalar_t* pos1 = &idata[h1 * input_width + w1];
        const scalar_t* pos2 = &odata[h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += input_width * input_height;
          pos2 += output_width * output_height;
        }
      }
    }
    return;
  }

  const scalar_t rheight = area_pixel_compute_scale<scalar_t>(
      input_height, output_height, align_corners, scales_h);
  const scalar_t rwidth = area_pixel_compute_scale<scalar_t>(
      input_width, output_width, align_corners, scales_w);

  for (int64_t h2 = 0; h2 < output_height; ++h2) {
    const scalar_t h1r = area_pixel_compute_source_index<scalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);

    const int64_t h1 = h1r;
    const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;

    const scalar_t h1lambda = h1r - h1;
    const scalar_t h0lambda = static_cast<scalar_t>(1.) - h1lambda;

    for (int64_t w2 = 0; w2 < output_width; ++w2) {
      const scalar_t w1r = area_pixel_compute_source_index<scalar_t>(
          rwidth, w2, align_corners, /*cubic=*/false);

      const int64_t w1 = w1r;
      const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;

      const scalar_t w1lambda = w1r - w1;
      const scalar_t w0lambda = static_cast<scalar_t>(1.) - w1lambda;

      scalar_t* pos1 = &idata[h1 * input_width + w1];

      const scalar_t* pos2 = &odata[h2 * output_width + w2];

      for (int64_t c = 0; c < channels; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * input_width] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * input_width + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += input_width * input_height;
        pos2 += output_width * output_height;
      }
    }
  }
}

static void upsample_bilinear2d_out_cpu_template(
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

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bilinear2d", [&] {
    auto* idata = input.data_ptr<scalar_t>();
    auto* odata = output.data_ptr<scalar_t>();

    upsample_bilinear2d_out_frame<scalar_t>(
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

static void upsample_bilinear2d_backward_out_cpu_template(
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
      grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();

        upsample_bilinear2d_backward_out_frame<scalar_t>(
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

Tensor& upsample_bilinear2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_bilinear2d_out_cpu_template(
      output, input, output_size, align_corners, scales_h, scales_w);
  return output;
}

Tensor upsample_bilinear2d_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  upsample_bilinear2d_out_cpu_template(
      output, input, output_size, align_corners, scales_h, scales_w);
  return output;
}

Tensor& upsample_bilinear2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_bilinear2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  return grad_input;
}

Tensor upsample_bilinear2d_backward_cpu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_bilinear2d_backward_out_cpu_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  return grad_input;
}

} // namespace native
} // namespace at
