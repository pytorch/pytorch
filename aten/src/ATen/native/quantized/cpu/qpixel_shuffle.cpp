#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/cpu/utils.h>

#include <c10/util/irange.h>

namespace at {
namespace native {

namespace {

Tensor quantized_pixel_shuffle_impl(
    const Tensor& self,
    int64_t upscale_factor) {

  TORCH_CHECK(
      upscale_factor > 0,
      "Number of upscale_factor to divide channels in must be positive.",
      " Value of upscale_factor:", upscale_factor);
  TORCH_CHECK(
      self.dim() == 4,
      "quantized_pixel_shuffle expects 4D input, but got input with sizes ",
      self.sizes());
  TORCH_CHECK(
      self.scalar_type() == kQUInt8,
      "quantized_pixel_shuffle works only on uint8_t.",
      "But got:", self.scalar_type());

  int64_t S = upscale_factor;
  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);
  int64_t height = self.size(2);
  int64_t width = self.size(3);
  int64_t out_channels = channels / (S * S);
  TORCH_CHECK(
      channels % (S * S) == 0,
      "quantized_pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
      "upscale_factor, but input.size(-3)=", channels, " is not divisible by ", (S * S));

  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::native::empty_affine_quantized(
      {nbatch, out_channels, height * S, width * S},
      kQUInt8,
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      self_nhwc.q_scale(),
      self_nhwc.q_zero_point(),
      MemoryFormat::ChannelsLast);

  // input strides
  int64_t stride_n = height * width * channels;
  int64_t stride_h = width * channels;
  int64_t stride_w = channels;
  int64_t stride_c = S * S;
  int64_t stride_s1 = S;
  int64_t stride_s2 = 1;

  auto qx_data = self_nhwc.data_ptr<c10::quint8>();
  auto qy_data = qy.data_ptr<c10::quint8>();

  // input tensor shape: [n, h, w, oc * S * S], channels = oc * S * S;
  // output tensor shape: [n, h * S, w * S, oc]
  //
  // parallel on both n and h dimension, use two steps to do the transformation
  at::parallel_for(0, nbatch * height, 0, [&](int64_t begin, int64_t end) {
    // thread local temp buffer
    std::unique_ptr<c10::quint8 []> buffer(new c10::quint8[channels]);

    for (const auto i : c10::irange(begin, end)) {
      int64_t n = i / height;
      int64_t h = i % height;
      for (const auto w : c10::irange(width)) {
        auto qx_ptr = qx_data + n * stride_n + h * stride_h + w * stride_w;

        // step 1: transpose each channel lane
        //   src: input channel view as [oc, s1*s2]
        //   dst: buffer view as [s1*s2, oc]
        utils::transpose<uint8_t>(
            out_channels,
            S * S,
            reinterpret_cast<uint8_t*>(qx_ptr),
            S * S, /* ld_src */
            reinterpret_cast<uint8_t*>(buffer.get()),
            out_channels /* ld_dst */);

        // step 2: copy from temp buffer to output lane
        //   src: buffer view as [s1, s2 * oc]
        //   dst: output channels view as [s1, w, s2 * oc]
        //   so we can loop on s1 and do memcpy of size s2 * oc
        for (const auto s1 : c10::irange(S)) {
          std::memcpy(
              qy_data + i * width * channels + s1 * width * S * out_channels + w * S * out_channels,
              buffer.get() + s1 * S * out_channels,
              S * out_channels * sizeof(typename c10::quint8::underlying));
        }
      }
    }
  });

  return qy;
}

} // namespace

Tensor quantized_pixel_shuffle_cpu(
    const Tensor& self,
    int64_t upscale_factor) {
  return quantized_pixel_shuffle_impl(self, upscale_factor);
}

} // namespace native
} // namespace at
