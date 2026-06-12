#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#endif

#include <cstring>

namespace at::native {
namespace {

// pre calculate interpolation params on width
struct UpsampleBilinearParamW {
  int64_t w1, w1p;
  float w0lambda, w1lambda;

  UpsampleBilinearParamW(int64_t w1, int64_t w1p, float w0lambda, float w1lambda)
    : w1(w1)
    , w1p(w1p)
    , w0lambda(w0lambda)
    , w1lambda(w1lambda) {}
};

// at::native functions for the native_functions.yaml
template <typename scalar_t>
void upsample_bilinear2d_out_frame(
    Tensor& output,
    const Tensor& input,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  auto* idata = static_cast<const scalar_t*>(input.const_data_ptr());
  auto* odata = static_cast<scalar_t*>(output.data_ptr());

  channels = channels * nbatch;
  if (channels == 0 || output_height == 0 || output_width == 0) {
    return;
  }
  auto* i_p = reinterpret_cast<const typename scalar_t::underlying*>(idata);
  auto* o_p = reinterpret_cast<typename scalar_t::underlying*>(odata);

  // special case: just copy
  if (input_height == output_height && input_width == output_width) {
    std::memcpy(
        o_p,
        i_p,
        channels * input_height * input_width *
            sizeof(typename scalar_t::underlying));
    return;
  }

  const auto rheight = area_pixel_compute_scale<float>(
      input_height, output_height, align_corners, scales_h);

  const auto rwidth = area_pixel_compute_scale<float>(
      input_width, output_width, align_corners, scales_w);

  float output_scale = static_cast<float>(output.q_scale() / input.q_scale());

  const int64_t input_q_zero_point = input.q_zero_point();
  const int64_t output_q_zero_point = output.q_zero_point();

  std::vector<UpsampleBilinearParamW> params_w;
  params_w.reserve(output_width);
  for (const auto w2 : c10::irange(output_width)) {
    const auto w1r = area_pixel_compute_source_index<float>(
        rwidth, w2, align_corners, /*cubic=*/false);

    const int64_t w1 = w1r;
    const int64_t w1p = (w1 < input_width - 1) ? 1 : 0;

    const float w1lambda = w1r - w1;
    const float w0lambda = static_cast<float>(1.) - w1lambda;

    params_w.emplace_back(w1, w1p, w0lambda, w1lambda);
  }

  // compared to 'nearest', each requires 4 points and takes additional * and +
  // set the scale to be 16.
  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, output_width) / 16;
  at::parallel_for(0, channels * output_height, grain_size, [&](int64_t begin, int64_t end) {
    int64_t nc{0}, h2{0};
    data_index_init(begin, nc, channels, h2, output_height);

    for (const auto i : c10::irange(begin, end)) {
      const auto h1r = area_pixel_compute_source_index<float>(
          rheight, h2, align_corners, /*cubic=*/false);

      const int64_t h1 = h1r;
      const int64_t h1p = (h1 < input_height - 1) ? 1 : 0;

      const float h1lambda = h1r - h1;
      const float h0lambda = static_cast<float>(1.) - h1lambda;

      const auto* i_ptr = &i_p[nc * input_height * input_width];
      auto* pos2 = &o_p[i * output_width];

      for (const auto w2 : c10::irange(output_width)) {
        const auto& param_w = params_w[w2];
        const int64_t w1 = param_w.w1;
        const int64_t w1p = param_w.w1p;
        const float w0lambda = param_w.w0lambda;
        const float w1lambda = param_w.w1lambda;

        const auto* pos1 = i_ptr + h1 * input_width + w1;

        const float result = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
            h1lambda *
                (w0lambda * pos1[h1p * input_width] +
                 w1lambda * pos1[h1p * input_width + w1p]) - input_q_zero_point;
        // requantization
        pos2[w2] = at::native::quantize_val<scalar_t>(
                      output_scale, output_q_zero_point, result)
                      .val_;
      }

      data_index_step(nc, channels, h2, output_height);
    }
  });

}

} // namespace

Tensor upsample_bilinear2d_quantized_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  auto full_output_size = native::upsample_2d_common_check(input.sizes(), output_size);

  TORCH_CHECK(
      input.numel() != 0 || c10::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  int64_t nbatch = full_output_size[0];
  int64_t channels = full_output_size[1];
  int64_t output_height = full_output_size[2];
  int64_t output_width = full_output_size[3];
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        std::nullopt);

    qupsample_bilinear2d_nhwc_stub(
        input.device().type(),
        output,
        input,
        input_height,
        input_width,
        output_height,
        output_width,
        nbatch,
        channels,
        align_corners,
        scales_h,
        scales_w);
    return output;
  } else {
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options(),
        input.q_scale(),
        input.q_zero_point());

    auto input_contig = input.contiguous();
    AT_DISPATCH_QINT_TYPES(
        input_contig.scalar_type(), "upsample_bilinear2d", [&] {
          upsample_bilinear2d_out_frame<scalar_t>(
              output,
              input_contig,
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
    return output;
  }
}

DEFINE_DISPATCH(qupsample_bilinear2d_nhwc_stub);
} // namespace at::native
