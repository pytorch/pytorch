#include <ATen/native/PixelShuffle.h>
#include <ATen/native/mps/OperationUtils.h>
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

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
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

  @autoreleasepool {
    string key = (upscale ? "pixel_shuffle_" : "pixel_unshuffle_") + getTensorsStringKey({self}) + "_factor_" +
        std::to_string(factor);
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if (!cachedGraph) {
      MPSCachedGraph* tmpCachedGraph = cache_->CreateCachedGraph(key, ^MPSCachedGraph*() {
        CachedGraph* newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          const auto ndims = self.ndimension();
          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor;
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
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph*>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        @{selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()};

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

Tensor pixel_shuffle_mps(const Tensor& self, int64_t upscale_factor) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: pixel_shuffle op is supported starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::native::pixel_shuffle_cpu(self.to("cpu"), upscale_factor).clone().to("mps");
  }

  return pixel_shuffle_helper(self, upscale_factor, /*upscale=*/true);
}

Tensor pixel_unshuffle_mps(const Tensor& self, int64_t downscale_factor) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: pixel_unshuffle op is supported starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::native::pixel_unshuffle_cpu(self.to("cpu"), downscale_factor).clone().to("mps");
  }

  return pixel_shuffle_helper(self, downscale_factor, /*upscale=*/false);
}

} // namespace at::native
