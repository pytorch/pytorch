#include <ATen/native/TensorTransformations.h>

#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

#include <ATen/native/cpu/PixelShuffleKernel.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace at {
namespace native {

Tensor _pixel_shuffle_cpu(const Tensor& self, IntArrayRef output_sizes, int64_t upscale_factor) {
  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);
  auto input = self.contiguous(memory_format);

  pixel_shuffle_kernel(kCPU, output, input, upscale_factor);
  return output;
}

Tensor _pixel_shuffle_backward_cpu(const Tensor& grad_output, IntArrayRef input_sizes, int64_t upscale_factor) {
  auto grad_input = at::empty({0}, grad_output.options());
  auto memory_format = grad_output.suggest_memory_format();
  grad_input.resize_(input_sizes, memory_format);
  auto grad_output_ = grad_output.contiguous(memory_format);

  pixel_shuffle_backward_kernel(kCPU, grad_input, grad_output_, upscale_factor);
  return grad_input;
}

Tensor _pixel_unshuffle_cpu(const Tensor& self, IntArrayRef output_sizes, int64_t downscale_factor) {
  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);
  auto input = self.contiguous(memory_format);

  pixel_unshuffle_kernel(kCPU, output, input, downscale_factor);
  return output;
}

Tensor _pixel_unshuffle_backward_cpu(const Tensor& grad_output, IntArrayRef input_sizes, int64_t downscale_factor) {
  auto grad_input = at::empty({0}, grad_output.options());
  auto memory_format = grad_output.suggest_memory_format();
  grad_input.resize_(input_sizes, memory_format);
  auto grad_output_ = grad_output.contiguous(memory_format);

  pixel_unshuffle_backward_kernel(kCPU, grad_input, grad_output_, downscale_factor);
  return grad_input;
}

Tensor pixel_shuffle(const Tensor& self, int64_t upscale_factor) {
  TORCH_CHECK(self.dim() >= 3,
              "pixel_shuffle expects input to have at least 3 dimensions, but got input with ",
              self.dim(), " dimension(s)");
  TORCH_CHECK(
      upscale_factor > 0,
      "pixel_shuffle expects a positive upscale_factor, but got ",
      upscale_factor);

  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  const auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  TORCH_CHECK(c % upscale_factor_squared == 0,
              "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
              "upscale_factor, but input.size(-3)=", c, " is not divisible by ", upscale_factor_squared);
  int64_t oc = c / upscale_factor_squared;
  int64_t oh = h * upscale_factor;
  int64_t ow = w * upscale_factor;

  // First, reshape to split the channels dim from c into 3 separate dims: (oc,
  // upscale_factor, upscale_factor). This allows shuffling to be done next by
  // permuting dims.
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {oc, upscale_factor, upscale_factor, h, w});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // Next, shuffle by permuting the new upscale_factor dims alongside the height and width dims.
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // std::iota is used to maintain the batch dims within the permutation.
  // Since 2 dims were added, the correct batch dim offsets are now:
  // -added_dims_shape.size(), ..., -7, -6.
  std::iota(permutation.begin(), permutation.end(), -added_dims_shape.size());
  permutation.insert(permutation.end(), {-5 /* oc */, -2 /* h */, -4 /* 1st upscale_factor */, -1 /* w */,
                                         -3 /* 2nd upscale_factor */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
  // and (w, upscale_factor) -> a single dim (ow).
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // Custome kernel for contiguous and channels last memory format on CPU
  if (self.device().type() == c10::DeviceType::CPU &&
      (self.scalar_type() == kFloat || self.scalar_type() == kDouble)) {
    return at::_pixel_shuffle_cpu(self, final_shape, upscale_factor);
  }

  return input_permuted.reshape(final_shape);
}


Tensor pixel_unshuffle(const Tensor& self, int64_t downscale_factor) {
  TORCH_CHECK(self.dim() >= 3,
              "pixel_unshuffle expects input to have at least 3 dimensions, but got input with ",
              self.dim(), " dimension(s)");
  TORCH_CHECK(
      downscale_factor > 0,
      "pixel_unshuffle expects a positive downscale_factor, but got ",
      downscale_factor);
  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  TORCH_CHECK(h % downscale_factor == 0,
             "pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=", h,
             " is not divisible by ", downscale_factor)
  TORCH_CHECK(w % downscale_factor == 0,
             "pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)=", w,
             " is not divisible by ", downscale_factor)
  int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  int64_t oc = c * downscale_factor_squared;
  int64_t oh = h / downscale_factor;
  int64_t ow = w / downscale_factor;

  // First, reshape to split height dim into (oh, downscale_factor) dims and
  // width dim into (ow, downscale_factor) dims. This allows unshuffling to be
  // done next by permuting dims.
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {c, oh, downscale_factor, ow, downscale_factor});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // Next, unshuffle by permuting the downscale_factor dims alongside the channel dim.
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // std::iota is used to maintain the batch dims within the permutation.
  // Since 2 dims were added, the correct batch dim offsets are now:
  // -added_dims_shape.size(), ..., -7, -6.
  std::iota(permutation.begin(), permutation.end(), -added_dims_shape.size());
  permutation.insert(permutation.end(), {-5 /* c */, -3 /* 1st downscale_factor */, -1 /*2nd downscale_factor */,
                                         -4 /* oh */, -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, downscale by collapsing (c, downscale_factor, downscale_factor) -> a single dim (oc),
  // resulting in height=oh and width=ow.
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // Custome kernel for contiguous and channels last memory format on CPU
  if (self.device().type() == c10::DeviceType::CPU &&
      (self.scalar_type() == kFloat || self.scalar_type() == kDouble)) {
    return at::_pixel_unshuffle_cpu(self, final_shape, downscale_factor);
  }

 return input_permuted.reshape(final_shape);
}

DEFINE_DISPATCH(pixel_shuffle_kernel);
DEFINE_DISPATCH(pixel_shuffle_backward_kernel);
DEFINE_DISPATCH(pixel_unshuffle_kernel);
DEFINE_DISPATCH(pixel_unshuffle_backward_kernel);

}} // namespace at::native
