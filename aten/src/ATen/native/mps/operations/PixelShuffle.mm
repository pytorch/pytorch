#include <ATen/native/PixelShuffle.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/pixel_shuffle_native.h>
#include <ATen/ops/pixel_unshuffle_native.h>

using namespace at::mps;

namespace at::native {

static Tensor pixel_shuffle_helper(const Tensor& self, int64_t factor, bool upscale) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (factor == 1) {
    return self.clone();
  }

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

  @autoreleasepool {
    std::string key = (upscale ? "pixel_shuffle_" : "pixel_unshuffle_") + getTensorsStringKey({self}) + "_factor_" +
        std::to_string(factor);
    CachedGraph* cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      const auto ndims = self.ndimension();
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* outputTensor = nullptr;
      if (upscale) {
        outputTensor = [mpsGraph depthToSpace2DTensor:inputTensor
                                            widthAxis:ndims - 1
                                           heightAxis:ndims - 2
                                            depthAxis:ndims - 3
                                            blockSize:factor
                                 usePixelShuffleOrder:YES
                                                 name:nil];
      } else {
        outputTensor = [mpsGraph spaceToDepth2DTensor:inputTensor
                                            widthAxis:ndims - 1
                                           heightAxis:ndims - 2
                                            depthAxis:ndims - 3
                                            blockSize:factor
                                 usePixelShuffleOrder:YES
                                                 name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      return newCachedGraph;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

Tensor pixel_shuffle_mps(const Tensor& self, int64_t upscale_factor) {
  return pixel_shuffle_helper(self, upscale_factor, /*upscale=*/true);
}

Tensor pixel_unshuffle_mps(const Tensor& self, int64_t downscale_factor) {
  return pixel_shuffle_helper(self, downscale_factor, /*upscale=*/false);
}

Tensor channel_shuffle_mps(const Tensor& self, int64_t groups) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(self.dim() > 2,
              "channel_shuffle expects input with > 2 dims, but got input with sizes ",
              self.sizes());

  const int64_t c = self.size(1);
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ",
              c, " channels and ", groups, " groups.");

  // Fast path for trivial cases
  if (groups == 1 || self.numel() == 0) {
    return self.clone();
  }

  MPSStream* stream = getCurrentMPSStream();

  Tensor output = at::empty_like(self, self.suggest_memory_format());

  @autoreleasepool {
    std::string key = "channel_shuffle_" + getTensorsStringKey({self}) + "_groups_" + std::to_string(groups);
    CachedGraph* cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      const int64_t ndim = self.dim();
      const int64_t channels = self.size(1);
      const int64_t channels_per_group = channels / groups;

      // Build the intermediate shape: (N, groups, channels_per_group, ...)
      // For input (N, C, H, W, ...) -> (N, groups, C/groups, H, W, ...)
      NSMutableArray<NSNumber*>* reshapeShape = [NSMutableArray arrayWithCapacity:ndim + 1];
      [reshapeShape addObject:@(self.size(0))];  // N
      [reshapeShape addObject:@(groups)];
      [reshapeShape addObject:@(channels_per_group)];
      for (int64_t i = 2; i < ndim; i++) {
        [reshapeShape addObject:@(self.size(i))];
      }

      // Reshape: (N, C, H, W) -> (N, groups, C/groups, H, W)
      MPSGraphTensor* reshapedTensor = [mpsGraph reshapeTensor:inputTensor
                                                     withShape:reshapeShape
                                                          name:nil];

      // Transpose dimensions 1 and 2: (N, groups, C/groups, H, W) -> (N, C/groups, groups, H, W)
      MPSGraphTensor* transposedTensor = [mpsGraph transposeTensor:reshapedTensor
                                                         dimension:1
                                                     withDimension:2
                                                              name:nil];

      // Build the output shape (same as input)
      NSMutableArray<NSNumber*>* outputShape = [NSMutableArray arrayWithCapacity:ndim];
      for (int64_t i = 0; i < ndim; i++) {
        [outputShape addObject:@(self.size(i))];
      }

      // Reshape back: (N, C/groups, groups, H, W) -> (N, C, H, W)
      MPSGraphTensor* outputTensor = [mpsGraph reshapeTensor:transposedTensor
                                                   withShape:outputShape
                                                        name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      return newCachedGraph;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

} // namespace at::native
