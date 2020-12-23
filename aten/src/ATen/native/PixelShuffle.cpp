#include <ATen/native/TensorTransformations.h>

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace at {
namespace native {

Tensor pixel_shuffle(const Tensor& self, int64_t upscale_factor) {
  TORCH_CHECK(self.dim() >= 3,
              "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
              self.dim(), " dimension(s)");
  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  const auto NUM_NON_BATCH_DIMS = 3;
  const auto last_batch_dim = self.sizes().end() - NUM_NON_BATCH_DIMS;

  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  TORCH_CHECK(c % upscale_factor_squared == 0,
              "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
              "upscale_factor, but input.size(-3)=", c, " is not divisible by ", upscale_factor_squared);
  int64_t oc = c / upscale_factor_squared;
  int64_t oh = h * upscale_factor;
  int64_t ow = w * upscale_factor;

  // First, reshape to expand the channels dim from c into 3 separate dims: (oc, upscale_factor, upscale_factor).
  // This allows shuffling to be done next by permuting dims.
  std::vector<int64_t> expanded_shape(self.sizes().begin(), last_batch_dim);
  expanded_shape.insert(expanded_shape.end(), {oc, upscale_factor, upscale_factor, h, w});
  const auto input_expanded = self.reshape(expanded_shape);

  // Next, shuffle by permuting the new upscale_factor dims alongside the height and width dims.
  std::vector<int64_t> permutation(self.sizes().begin(), last_batch_dim);
  // std::iota is used to maintain the batch dims within the permutation.
  // Since expansion added 2 dims, the correct batch dim offsets are now: -expanded_shape.size(), ..., -7, -6.
  std::iota(permutation.begin(), permutation.end(), -expanded_shape.size());
  permutation.insert(permutation.end(), {-5 /* oc */, -2 /* h */, -4 /* 1st upscale_factor */, -1 /* w */,
                                         -3 /* 2nd upscale_factor */});
  const auto input_permuted = input_expanded.permute(permutation);

  // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
  // and (w, upscale_factor) -> a single dim (ow).
  std::vector<int64_t> final_shape(self.sizes().begin(), last_batch_dim);
  final_shape.insert(final_shape.end(), {oc, oh, ow});
  return input_permuted.reshape(final_shape);
}

}} // namespace at::native
