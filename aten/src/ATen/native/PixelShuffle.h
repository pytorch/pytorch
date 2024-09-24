#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

inline void check_pixel_shuffle_shapes(const Tensor& self, int64_t upscale_factor) {
  TORCH_CHECK(self.dim() >= 3,
              "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
              self.dim(), " dimension(s)");
  TORCH_CHECK(upscale_factor > 0,
              "pixel_shuffle expects a positive upscale_factor, but got ",
              upscale_factor);
  int64_t c = self.size(-3);
  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  TORCH_CHECK(c % upscale_factor_squared == 0,
              "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
              "upscale_factor, but input.size(-3)=", c, " is not divisible by ", upscale_factor_squared);
}

inline void check_pixel_unshuffle_shapes(const Tensor& self, int64_t downscale_factor) {
  TORCH_CHECK(
      self.dim() >= 3,
      "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
      self.dim(),
      " dimension(s)");
  TORCH_CHECK(
      downscale_factor > 0,
      "pixel_unshuffle expects a positive downscale_factor, but got ",
      downscale_factor);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  TORCH_CHECK(
      h % downscale_factor == 0,
      "pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=",
      h,
      " is not divisible by ",
      downscale_factor);
  TORCH_CHECK(
      w % downscale_factor == 0,
      "pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)=",
      w,
      " is not divisible by ",
      downscale_factor);
}

}} // namespace at::native
