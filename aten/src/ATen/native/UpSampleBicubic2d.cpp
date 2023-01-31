#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>
#include <ATen/Parallel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bicubic2d_aa.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/upsample_bicubic2d.h>
#include <ATen/ops/upsample_bicubic2d_backward.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#include <ATen/ops/upsample_bicubic2d_native.h>
#endif

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

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
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

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

TORCH_META_FUNC(_upsample_bicubic2d_aa) (
  const Tensor& input, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w
) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

TORCH_META_FUNC(_upsample_bicubic2d_aa_backward) (
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

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(", i, ") = ", full_output_size[i],
        " but got grad_output.size(", i, ") = ", grad_output.size(i));
  }

  set_output_raw_strided(0, input_size, {}, grad_output.options());
}

} // namespace meta
namespace native {
namespace {

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
  auto input_slice_size = input_height * input_width;
  auto output_slice_size = output_height * output_width;

  using opmath_t = at::opmath_type<scalar_t>;
  const opmath_t height_scale = area_pixel_compute_scale<opmath_t>(
      input_height, output_height, align_corners, scales_h);
  const opmath_t width_scale = area_pixel_compute_scale<opmath_t>(
      input_width, output_width, align_corners, scales_w);
  at::parallel_for(0, channels, at::internal::GRAIN_SIZE / output_slice_size / 4, [&](int64_t start, int64_t end) {
    opmath_t* acc_data_ptr = nullptr;
    std::unique_ptr<opmath_t[]> buffer_data;
    if constexpr (!std::is_same<scalar_t, opmath_t>::value) {
      buffer_data = std::make_unique<opmath_t[]>(input_slice_size);
      acc_data_ptr = buffer_data.get();
      memset(acc_data_ptr, 0, sizeof(opmath_t) * input_slice_size);
    }
    for (const auto i : c10::irange(start, end)) {
      scalar_t* in = idata + i * input_slice_size;
      scalar_t* out = odata + i * output_slice_size;
      for (const auto output_y : c10::irange(output_height)) {
        for (const auto output_x : c10::irange(output_width)) {

          const opmath_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
          int64_t input_x;
          opmath_t t_x;
          guard_index_and_lambda(real_x, input_width, input_x, t_x);

          const opmath_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
          int64_t input_y;
          opmath_t t_y;
          guard_index_and_lambda(real_y, input_height, input_y, t_y);

          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
          opmath_t x_coeffs[4];
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
          opmath_t y_coeffs[4];

          get_cubic_upsample_coefficients<opmath_t>(x_coeffs, t_x);
          get_cubic_upsample_coefficients<opmath_t>(y_coeffs, t_y);

          opmath_t out_value = out[output_y * output_width + output_x];
          for (const auto ii : c10::irange(4)) {
            for (const auto jj : c10::irange(4)) {
              upsample_increment_value_bounded<opmath_t>(
                  acc_data_ptr == nullptr ? reinterpret_cast<opmath_t*>(in) : acc_data_ptr,
                  input_width,
                  input_height,
                  input_x - 1 + ii,
                  input_y - 1 + jj,
                  out_value * y_coeffs[jj] * x_coeffs[ii]);
            }
          }
        }
      }
      if (acc_data_ptr != nullptr) {
        apply_grad_input(acc_data_ptr, in, input_slice_size);
      }
    }
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
  // Special case: input/output same size, just copy
  if (input_height == output_height && input_width == output_width) {
    grad_input.copy_(grad_output);
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
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
  upsample_bicubic2d_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
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

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_out_cpu) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output
) {
  _upsample_bicubic2d_aa_kernel(kCPU, output, input, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_backward_out_cpu) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input
) {
  grad_input.zero_();
  _upsample_bicubic2d_aa_backward_kernel(kCPU, grad_input, grad_output, align_corners, scales_h, scales_w);
}

// vec variants

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_bicubic2d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_bicubic2d(input, osize, align_corners, scale_h, scale_w);
}

Tensor _upsample_bicubic2d_aa(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::_upsample_bicubic2d_aa(input, osize, align_corners, scale_h, scale_w);
}

DEFINE_DISPATCH(upsample_bicubic2d_kernel);
DEFINE_DISPATCH(_upsample_bicubic2d_aa_kernel);
DEFINE_DISPATCH(_upsample_bicubic2d_aa_backward_kernel);

} // namespace native
} // namespace at
