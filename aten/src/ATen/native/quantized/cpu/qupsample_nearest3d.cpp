#include <ATen/ATen.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/util/irange.h>

#include <cstring>


namespace at {
namespace native {

// at::native functions for the native_functions.yaml
template <typename scalar_t>
static void upsample_nearest3d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  channels = channels * nbatch;
  auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata);
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);

  // special case: just copy
  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    std::memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename scalar_t::underlying));
    return;
  }

  for (int64_t d2 = 0; d2 < output_depth; ++d2) {
    const int64_t d1 =
          nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);

    for (int64_t h2 = 0; h2 < output_height; ++h2) {
      const int64_t h1 =
          nearest_neighbor_compute_source_index(height_scale, h2, input_height);

      for (int64_t w2 = 0; w2 < output_width; ++w2) {
        const int64_t w1 =
            nearest_neighbor_compute_source_index(width_scale, w2, input_width);

        const auto* pos1 = &i_p[d1 * input_height * input_width + h1 * input_width + w1];
        auto* pos2 = &o_p[d2 * output_height * output_width + h2 * output_width + w2];

        for (int64_t c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += input_depth * input_height * input_width;
          pos2 += output_depth * output_height * output_width;
        }
      }
    }
  }
}

template <typename scalar_t>
static void upsample_nearest3d_out_frame_nhwc(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
  float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  for (const auto b : c10::irange(nbatch)) {
    auto* i_p = reinterpret_cast<typename scalar_t::underlying*>(idata + b * input_depth * input_height * input_width * channels);
    auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata + b * output_depth * output_height * output_width * channels);
    // special case: just copy
    if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
      std::memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename scalar_t::underlying));
      return;
    }

    for (int64_t d2 = 0; d2 < output_depth; ++d2) {
      const int64_t d1 =
          nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);
      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        const int64_t h1 =
            nearest_neighbor_compute_source_index(height_scale, h2, input_height);

        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          const int64_t w1 =
              nearest_neighbor_compute_source_index(width_scale, w2, input_width);

          const auto* pos1 = &i_p[(d1 * input_height * input_width + h1 * input_width + w1)*channels];
          auto* pos2 = &o_p[(d2 * output_height * output_width + h2 * output_width + w2)*channels];
          std::memcpy(pos2, pos1, channels * sizeof(typename scalar_t::underlying));
        }
      }
    }
  }
}

Tensor upsample_nearest3d_quantized_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input.numel() != 0 && input.dim() == 5,
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);
  AT_ASSERT(input_width > 0 && output_width > 0);
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_depth, output_height, output_width},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);

    AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_nearest3d", [&] {
      auto* idata = static_cast<scalar_t*>(input.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest3d_out_frame_nhwc<scalar_t>(
          odata,
          idata,
          input_depth,
          input_height,
          input_width,
          output_depth,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_d,
          scales_h,
          scales_w);
    });
    return output;
  } else {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_depth, output_height, output_width},
        input.options(),
        input.q_scale(),
        input.q_zero_point());

    auto input_contig = input.contiguous();

    AT_DISPATCH_QINT_TYPES(input_contig.scalar_type(), "upsample_nearest3d", [&] {
      auto* idata = static_cast<scalar_t*>(input_contig.data_ptr());
      auto* odata = static_cast<scalar_t*>(output.data_ptr());
      upsample_nearest3d_out_frame<scalar_t>(
          odata,
          idata,
          input_depth,
          input_height,
          input_width,
          output_depth,
          output_height,
          output_width,
          nbatch,
          channels,
          scales_d,
          scales_h,
          scales_w);
    });
    return output;
  }
}

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor upsample_nearest3d_quantized_cpu(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return upsample_nearest3d_quantized_cpu(input, osize, scale_d, scale_h, scale_w);
}

} // namespace native
} // namespace at
