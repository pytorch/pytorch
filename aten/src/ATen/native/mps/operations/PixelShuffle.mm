#include <ATen/native/PixelShuffle.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/pixel_shuffle_native.h>
#include <ATen/ops/pixel_unshuffle_native.h>

using namespace at::mps;

namespace at::native {

Tensor pixel_shuffle_mps(const Tensor& self, int64_t upscale_factor) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: pixel_shuffle op is supported starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::native::pixel_shuffle_cpu(self.to("cpu")).clone().to("mps");
  }

  if (upscale_factor == 1) {
    return self;
  }

  check_pixel_shuffle_shapes(self, upscale_factor);

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  MPSStream* stream = getCurrentMPSStream();

  const int64_t c = self.size(-3);
  const int64_t h = self.size(-2);
  const int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  const int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  const int64_t oc = c / upscale_factor_squared;
  const int64_t oh = h * upscale_factor;
  const int64_t ow = w * upscale_factor;

  std::vector<int64_t> out_shape(self.sizes().begin(), self_sizes_batch_end);
  out_shape.insert(out_shape.end(), {oc, oh, ow});

  Tensor output = at::empty(out_shape, self.options());

  @autoreleasepool {
    string key = "pixel_shuffle_" + getTensorsStringKey({self}) + "_upscale_factor_" + std::to_string(upscale_factor);
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if (!cachedGraph) {
      MPSCachedGraph* tmpCachedGraph = cache_->CreateCachedGraph(key, ^MPSCachedGraph*() {
        CachedGraph* newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          const auto ndims = self.ndimension();
          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = [mpsGraph depthToSpace2DTensor:inputTensor
                                                              widthAxis:ndims - 1
                                                             heightAxis:ndims - 2
                                                              depthAxis:ndims - 3
                                                              blockSize:upscale_factor
                                                   usePixelShuffleOrder:YES
                                                                   name:nil];

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

Tensor pixel_unshuffle_mps(const Tensor& self, int64_t downscale_factor) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: pixel_unshuffle op is supported starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::native::pixel_unshuffle_cpu(self.to("cpu")).clone().to("mps");
  }

  if (downscale_factor == 1) {
    return self;
  }

  check_pixel_unshuffle_shapes(self, downscale_factor);

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  MPSStream* stream = getCurrentMPSStream();

  const int64_t c = self.size(-3);
  const int64_t h = self.size(-2);
  const int64_t w = self.size(-1);
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  const int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  const int64_t oc = c * downscale_factor_squared;
  const int64_t oh = h / downscale_factor;
  const int64_t ow = w / downscale_factor;

  std::vector<int64_t> out_shape(self.sizes().begin(), self_sizes_batch_end);
  out_shape.insert(out_shape.end(), {oc, oh, ow});

  Tensor output = at::empty(out_shape, self.options());

  @autoreleasepool {
    string key =
        "pixel_unshuffle_" + getTensorsStringKey({self}) + "_downscale_factor_" + std::to_string(downscale_factor);
    CachedGraph* cachedGraph = cache_->LookUpAs<CachedGraph>(key);
    if (!cachedGraph) {
      MPSCachedGraph* tmpCachedGraph = cache_->CreateCachedGraph(key, ^MPSCachedGraph*() {
        CachedGraph* newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          const auto ndims = self.ndimension();
          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = [mpsGraph spaceToDepth2DTensor:inputTensor
                                                              widthAxis:ndims - 1
                                                             heightAxis:ndims - 2
                                                              depthAxis:ndims - 3
                                                              blockSize:downscale_factor
                                                   usePixelShuffleOrder:YES
                                                                   name:nil];

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

} // namespace at::native
