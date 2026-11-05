#include <ATen/native/PixelShuffle.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/pixel_shuffle_native.h>
#include <ATen/ops/pixel_unshuffle_native.h>
#include <c10/util/safe_conv.h>

using namespace at::mps;

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/PixelShuffle_metallib.h>
#endif

static Tensor pixel_shuffle_helper(const Tensor& self, int64_t factor, bool upscale) {
  using namespace mps;

  if (upscale) {
    check_pixel_shuffle_shapes(self, factor);
  } else {
    check_pixel_unshuffle_shapes(self, factor);
  }

  MPSStream* stream = getCurrentMPSStream();

  const int64_t c = self.size(-3);
  const int64_t h = self.size(-2);
  const int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  const int64_t factor_squared = factor * factor;
  const int64_t oc = upscale ? c / factor_squared : c * factor_squared;
  const int64_t oh = upscale ? h * factor : h / factor;
  const int64_t ow = upscale ? w * factor : w / factor;

  std::vector<int64_t> out_shape(self.sizes().begin(), self_sizes_batch_end);
  out_shape.insert(out_shape.end(), {oc, oh, ow});

  Tensor output = at::empty(out_shape, self.options());

  if (output.numel() == 0) {
    return output;
  }

  // Strides of a depth pixel (n, c, hd, wd) followed by the steps of its f*f sub-pixels (i, j): the depth layout
  // (N, C*f*f, Hd, Wd) packs the sub-pixels into channels, the space layout (N, C, Hd*f, Wd*f) spreads them over h, w.
  const auto fold_strides = [factor, factor_squared](const Tensor& tensor, bool is_depth) {
    const int64_t n_stride = tensor.stride(0), c_stride = tensor.stride(1), h_stride = tensor.stride(2),
                  w_stride = tensor.stride(3);
    return is_depth
        ? std::array<int64_t, 6>{n_stride, factor_squared * c_stride, h_stride, w_stride, factor * c_stride, c_stride}
        : std::array<int64_t, 6>{n_stride, c_stride, factor * h_stride, factor * w_stride, h_stride, w_stride};
  };
  const Tensor input = self.reshape({-1, c, h, w});
  const auto in_strides = fold_strides(input, upscale);
  const auto out_strides = fold_strides(output.view({-1, oc, oh, ow}), !upscale);
  const Tensor& depth = upscale ? input : output;
  const int64_t hd = depth.size(-2), wd = depth.size(-1), channels = upscale ? oc : c;
  // channels_last input: walk c along the fastest thread axis so the reads coalesce.
  const bool channel_major = input.stride(1) < input.stride(3);
  // Grid (wd, hd, n * C + c), or (c, wd, n * Hd + hd) when channel_major. Keeping hd and wd on separate axes lets a
  // depth plane with 2^32 or more pixels fit the 32-bit per-axis thread ids.
  const int64_t z_dim = channel_major ? hd : channels;
  const std::array<uint32_t, 2> factor_zdim = {c10::safe_conv<uint32_t>(factor, "factor"),
                                               c10::safe_conv<uint32_t>(z_dim, "z_dim")};
  const std::string kernel = "pixel_shuffle_" + std::to_string(input.element_size());
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, upscale ? "pixel_shuffle" : "pixel_unshuffle", {input}, stream);
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, input, output, in_strides, out_strides, factor_zdim, channel_major);
      mtl_dispatch3DJob(
          computeEncoder, pso, channel_major ? channels : wd, channel_major ? wd : hd, input.size(0) * z_dim);
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  return output;
}

Tensor pixel_shuffle_mps(const Tensor& self, int64_t upscale_factor) {
  return pixel_shuffle_helper(self, upscale_factor, /*upscale=*/true);
}

Tensor pixel_unshuffle_mps(const Tensor& self, int64_t downscale_factor) {
  return pixel_shuffle_helper(self, downscale_factor, /*upscale=*/false);
}

} // namespace at::native
