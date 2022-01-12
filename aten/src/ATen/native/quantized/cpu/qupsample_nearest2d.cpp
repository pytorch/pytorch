#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/UpSample.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#endif

#include <c10/util/irange.h>

#include <cstring>


namespace at {
namespace native {

// Define a typedef to dispatch to nearest_idx or nearest_exact_idx
typedef int64_t (*nn_compute_source_index_fn_t)(const float, int64_t, int64_t);

// at::native functions for the native_functions.yaml
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  channels = channels * nbatch;
  if (channels == 0 || output_height == 0 || output_width == 0) {
    return;
  }
  auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata);
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);

  // special case: just copy
  if (input_height == output_height && input_width == output_width) {
    std::memcpy(o_p, i_p, channels * input_height * input_width * sizeof(typename scalar_t::underlying));
    return;
  }

  for (const auto h2 : c10::irange(output_height)) {
    const int64_t h1 =
        nn_compute_source_index_fn(height_scale, h2, input_height);

    for (const auto w2 : c10::irange(output_width)) {
      const int64_t w1 =
          nn_compute_source_index_fn(width_scale, w2, input_width);

      const auto* pos1 = &i_p[h1 * input_width + w1];
      auto* pos2 = &o_p[h2 * output_width + w2];

      for (const auto c : c10::irange(channels)) {
        (void)c; //Suppress unused variable warning
        pos2[0] = pos1[0];
        pos1 += input_height * input_width;
        pos2 += output_height * output_width;
      }
    }
  }
}

template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_frame_nhwc(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  for (const auto b : c10::irange(nbatch)) {
    auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata + b * input_height * input_width * channels);
    auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata + b * output_height * output_width * channels);
    // special case: just copy
    if (input_height == output_height && input_width == output_width) {
      std::memcpy(o_p, i_p, channels * input_height * input_width * sizeof(typename scalar_t::underlying));
      return;
    }

    for (const auto h2 : c10::irange(output_height)) {
      const int64_t h1 =
          nn_compute_source_index_fn(height_scale, h2, input_height);

      for (const auto w2 : c10::irange(output_width)) {
        const int64_t w1 =
            nn_compute_source_index_fn(width_scale, w2, input_width);

        const auto* pos1 = &i_p[(h1 * input_width + w1)*channels];
        auto* pos2 = &o_p[(h2 * output_width + w2)*channels];
        std::memcpy(pos2, pos1, channels * sizeof(typename scalar_t::underlying));
      }
    }
  }
}

template <nn_compute_source_index_fn_t nn_compute_source_index_fn>
Tensor _upsample_nearest2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input.dim() == 4,
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
    AT_ASSERT(input_width > 0 && output_width > 0);
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);

    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_nearest2d", [&] {
      auto* idata = static_cast<scalar_t*>(input.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest2d_out_frame_nhwc<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_height,
          input_width,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_h,
          scales_w);
    });
    return output;
  } else {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options(),
        input.q_scale(),
        input.q_zero_point());

    auto input_contig = input.contiguous();

    AT_DISPATCH_QINT_TYPES(input_contig.scalar_type(), "upsample_nearest2d", [&] {
      auto* idata = static_cast<scalar_t*>(input_contig.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest2d_out_frame<scalar_t, nn_compute_source_index_fn>(
          odata,
          idata,
          input_height,
          input_width,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_h,
          scales_w);
    });
    return output;
  }
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_nearest2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef osize,
    c10::optional<double> scale_h,
    c10::optional<double> scale_w) {
  return _upsample_nearest2d_quantized_cpu<nearest_neighbor_compute_source_index>(input, osize, scale_h, scale_w);
}

Tensor _upsample_nearest_exact2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef osize,
    c10::optional<double> scale_h,
    c10::optional<double> scale_w) {
  return _upsample_nearest2d_quantized_cpu<nearest_neighbor_exact_compute_source_index>(input, osize, scale_h, scale_w);
}

Tensor upsample_nearest2d_quantized_cpu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return upsample_nearest2d_quantized_cpu(input, osize, scale_h, scale_w);
}

Tensor _upsample_nearest_exact2d_quantized_cpu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return _upsample_nearest_exact2d_quantized_cpu(input, osize, scale_h, scale_w);
}

} // namespace native
} // namespace at
