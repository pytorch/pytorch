#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>

namespace at {
namespace native {

void grid_sampler_2d_mps_impl(Tensor &output, const Tensor& input, const Tensor& grid,
                              int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners) {
// Grid Sampler support has been added in macOS 13.1
#if defined(__MAC_13_2)
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  MPSGraphResizeMode samplingMode;
  MPSGraphPaddingMode paddingMode;

  auto memory_format = input.suggest_memory_format();
  MPSGraphTensorNamedDataLayout inputTensorLayout =
      (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW : MPSGraphTensorNamedDataLayoutNHWC;

  switch (static_cast<GridSamplerPadding>(padding_mode)) {
    case GridSamplerPadding::Zeros:
      paddingMode = MPSGraphPaddingModeZero; break;
    case GridSamplerPadding::Border:
      TORCH_CHECK(false, "MPS: Unsupported Border padding mode"); break;
    case GridSamplerPadding::Reflection:
      paddingMode = align_corners == true ? MPSGraphPaddingModeReflect : MPSGraphPaddingModeSymmetric; break;
    default:
      TORCH_CHECK(false, "MPS: Unrecognised Padding Mode: ", padding_mode);
  }

  switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
    case GridSamplerInterpolation::Bilinear:
      samplingMode = MPSGraphResizeBilinear; break;
    case GridSamplerInterpolation::Nearest:
      samplingMode = MPSGraphResizeNearest; break;
    case GridSamplerInterpolation::Bicubic:
      TORCH_CHECK(false, "MPS: Unsupported Bicubic interpolation"); break;
    default:
      TORCH_CHECK(false, "MPS: Unrecognised interpolation mode: ", interpolation_mode); break;
   }

  MPSStream *stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gridTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "grid_sampler_2d_mps"                     +
                  getTensorsStringKey({input, grid})       +
                  ":" + std::to_string(interpolation_mode) +
                  ":" + std::to_string(padding_mode)       +
                  ":" + std::to_string(align_corners);

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
          MPSGraphTensor* gridTensor = mpsGraphRankedPlaceHolder(mpsGraph, grid);

          MPSGraphTensor* outputTensor = nil;
          if (static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Nearest) {
            outputTensor = [mpsGraph sampleGridWithSourceTensor: inputTensor
                                               coordinateTensor: gridTensor
                                                         layout: inputTensorLayout
                                           normalizeCoordinates: TRUE
                                            relativeCoordinates: FALSE
                                                   alignCorners: align_corners
                                                    paddingMode: paddingMode
                                            nearestRoundingMode: MPSGraphResizeNearestRoundingModeRoundToEven
                                                  constantValue: 0.0f
                                                           name: nil];
          } else {
            outputTensor = [mpsGraph sampleGridWithSourceTensor: inputTensor
                                               coordinateTensor: gridTensor
                                                         layout: inputTensorLayout
                                           normalizeCoordinates: TRUE
                                            relativeCoordinates: FALSE
                                                   alignCorners: align_corners
                                                    paddingMode: paddingMode
                                                   samplingMode: samplingMode
                                                  constantValue: 0.0f
                                                           name: nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gridTensor_ = gridTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder gridPlaceholder = Placeholder(cachedGraph->gridTensor_, grid);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);


    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      gridPlaceholder.getMPSGraphTensor() : gridPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
#endif // defined(__MAC_13_2)
}

Tensor grid_sampler_2d_mps(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
#if defined(__MAC_13_2)
  bool xcode_sdk_13_2_or_higher = true;
#else
  bool xcode_sdk_13_2_or_higher = false;
#endif

  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS) || !xcode_sdk_13_2_or_higher) {
    TORCH_WARN_ONCE("MPS: grid_sampler_2d op is supported natively starting from macOS 13.1. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::grid_sampler_2d(
      input.to("cpu"), grid.to("cpu"), interpolation_mode, padding_mode, align_corners).clone().to("mps");
  }

  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());

  grid_sampler_2d_mps_impl(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

} // namespace native
} // namespace at
