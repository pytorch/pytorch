#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorTransformations.h>
#include <ATen/native/cpu/PixelShuffleKernel.h>
#include <ATen/native/PixelShuffle.h>

#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/pixel_shuffle_native.h>
#include <ATen/ops/pixel_unshuffle_native.h>
#endif

#include <algorithm>
#include <numeric>
#include <vector>

namespace at {
namespace native {

Tensor pixel_shuffle_cpu(const Tensor& self, int64_t upscale_factor) {
  check_pixel_shuffle_shapes(self, upscale_factor);

  // Format: (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(output_sizes.end(),
      {self.size(-3) / upscale_factor / upscale_factor,
       self.size(-2) * upscale_factor,
       self.size(-1) * upscale_factor});

  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);

  if (output.numel() == 0) {
    return output;
  }

  auto input = self.contiguous(memory_format);

  pixel_shuffle_kernel(kCPU, output, input, upscale_factor);
  return output;
}

Tensor pixel_unshuffle_cpu(const Tensor& self, int64_t downscale_factor) {
  check_pixel_unshuffle_shapes(self, downscale_factor);

  if (self.numel() == 0) {
    return self.clone();
  }

  // Format: (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(output_sizes.end(),
      {self.size(-3) * downscale_factor * downscale_factor,
       self.size(-2) / downscale_factor,
       self.size(-1) / downscale_factor});

  auto output = at::empty({0}, self.options());
  auto memory_format = self.suggest_memory_format();
  output.resize_(output_sizes, memory_format);

  if (output.numel() == 0) {
    return output;
  }

  auto input = self.contiguous(memory_format);

  pixel_unshuffle_kernel(kCPU, output, input, downscale_factor);
  return output;
}

Tensor math_pixel_shuffle(const Tensor& self, int64_t upscale_factor) {
  check_pixel_shuffle_shapes(self, upscale_factor);

  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  const auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
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
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* oc */, -2 /* h */, -4 /* 1st upscale_factor */, -1 /* w */,
                                         -3 /* 2nd upscale_factor */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, upscale by collapsing (h, upscale_factor) -> a single dim (oh)
  // and (w, upscale_factor) -> a single dim (ow).
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // pixel_shuffle expects to *never* return an alias of the input.
  return input_permuted.clone(at::MemoryFormat::Contiguous).view(final_shape);
}

Tensor math_pixel_unshuffle(const Tensor& self, int64_t downscale_factor) {
  check_pixel_unshuffle_shapes(self, downscale_factor);

  // Format: (B1, ..., Bn), C, H, W
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

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
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* c */, -3 /* 1st downscale_factor */, -1 /*2nd downscale_factor */,
                                         -4 /* oh */, -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // Finally, downscale by collapsing (c, downscale_factor, downscale_factor) -> a single dim (oc),
  // resulting in height=oh and width=ow.
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // pixel_unshuffle expects to *never* return an alias of the input.
  return input_permuted.clone(at::MemoryFormat::Contiguous).view(final_shape);
}

DEFINE_DISPATCH(pixel_shuffle_kernel);
DEFINE_DISPATCH(pixel_unshuffle_kernel);

}} // namespace at::native
