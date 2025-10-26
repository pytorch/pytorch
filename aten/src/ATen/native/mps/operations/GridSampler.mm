#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GridSampler_metallib.h>
#endif

namespace mps {
static void grid_sampler_2d_mps_impl(Tensor& output,
                                     const Tensor& input,
                                     const Tensor& grid,
                                     int64_t interpolation_mode,
                                     int64_t padding_mode,
                                     bool align_corners) {
  // Grid Sampler support has been added in macOS 13.2
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  MPSGraphResizeMode samplingMode;
  MPSGraphPaddingMode paddingMode;

  auto memory_format = input.suggest_memory_format();
  MPSGraphTensorNamedDataLayout inputTensorLayout = (memory_format == at::MemoryFormat::Contiguous)
      ? MPSGraphTensorNamedDataLayoutNCHW
      : MPSGraphTensorNamedDataLayoutNHWC;

  switch (static_cast<GridSamplerPadding>(padding_mode)) {
    case GridSamplerPadding::Zeros:
      paddingMode = MPSGraphPaddingModeZero;
      break;
    case GridSamplerPadding::Border:
      TORCH_CHECK(false, "MPS: Unsupported Border padding mode");
      break;
    case GridSamplerPadding::Reflection:
      paddingMode = align_corners == true ? MPSGraphPaddingModeReflect : MPSGraphPaddingModeSymmetric;
      break;
    default:
      TORCH_CHECK(false, "MPS: Unrecognised Padding Mode: ", padding_mode);
  }

  switch (static_cast<GridSamplerInterpolation>(interpolation_mode)) {
    case GridSamplerInterpolation::Bilinear:
      samplingMode = MPSGraphResizeBilinear;
      break;
    case GridSamplerInterpolation::Nearest:
      samplingMode = MPSGraphResizeNearest;
      break;
    case GridSamplerInterpolation::Bicubic:
      TORCH_CHECK(false, "MPS: Unsupported Bicubic interpolation");
      break;
    default:
      TORCH_CHECK(false, "MPS: Unrecognised interpolation mode: ", interpolation_mode);
      break;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gridTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()), "foobar");
  @autoreleasepool {
    std::string key = "grid_sampler_2d_mps" + getTensorsStringKey({input, grid}) + ":" +
        std::to_string(interpolation_mode) + ":" + std::to_string(padding_mode) + ":" + std::to_string(align_corners);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* gridTensor = mpsGraphRankedPlaceHolder(mpsGraph, grid);

      MPSGraphTensor* outputTensor = nil;
      if (static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Nearest) {
        outputTensor = [mpsGraph sampleGridWithSourceTensor:inputTensor
                                           coordinateTensor:gridTensor
                                                     layout:inputTensorLayout
                                       normalizeCoordinates:TRUE
                                        relativeCoordinates:FALSE
                                               alignCorners:align_corners
                                                paddingMode:paddingMode
#if defined(__MAC_13_2)
                                        nearestRoundingMode:MPSGraphResizeNearestRoundingModeRoundToEven
#else
                                        nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)4L
#endif
                                              constantValue:0.0f
                                                       name:nil];
      } else {
        outputTensor = [mpsGraph sampleGridWithSourceTensor:inputTensor
                                           coordinateTensor:gridTensor
                                                     layout:inputTensorLayout
                                       normalizeCoordinates:TRUE
                                        relativeCoordinates:FALSE
                                               alignCorners:align_corners
                                                paddingMode:paddingMode
                                               samplingMode:samplingMode
                                              constantValue:0.0f
                                                       name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gridTensor_ = gridTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder gridPlaceholder = Placeholder(cachedGraph->gridTensor_, grid);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(inputPlaceholder, gridPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void grid_sampler_template(Tensor& output,
                                  const Tensor& input,
                                  const Tensor& grid,
                                  int64_t _interpolation_mode,
                                  int64_t _padding_mode,
                                  bool align_corners,
                                  int32_t sampler_dims,
                                  const std::string& op_name) {
  check_grid_sampler_common(input, grid);
  switch (sampler_dims) {
    case 2:
      check_grid_sampler_2d(input, grid);
      break;
    case 3:
      check_grid_sampler_3d(input, grid, _interpolation_mode);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Only 2D and 3D sampling are supported, but got: ", sampler_dims);
  }
  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  auto interpolation_mode = static_cast<GridSamplerInterpolation>(_interpolation_mode);
  auto padding_mode = static_cast<GridSamplerPadding>(_padding_mode);

  switch (interpolation_mode) {
    case GridSamplerInterpolation::Bilinear:
      break;
    case GridSamplerInterpolation::Nearest:
      TORCH_CHECK(false, op_name, ": Unsupported Nearest interpolation");
      break;
    case GridSamplerInterpolation::Bicubic:
      TORCH_CHECK(false, op_name, ": Unsupported Bicubic interpolation");
      break;
    default:
      TORCH_CHECK(false, op_name, ": Unrecognised interpolation mode: ", _interpolation_mode);
  }

  switch (padding_mode) {
    case GridSamplerPadding::Zeros:
    case GridSamplerPadding::Border:
    case GridSamplerPadding::Reflection:
      break;
    default:
      TORCH_CHECK(false, op_name, ": Unrecognised Padding Mode: ", _padding_mode);
  }

  auto input_size = input.sizes();
  auto grid_size = grid.sizes();
  output.resize_({input_size[0], input_size[1], grid_size[1], grid_size[2], grid_size[3]}, MemoryFormat::Contiguous);

  auto dims = input.dim();

  GridSamplerParams<5> params;
  params.sampler_dims = sampler_dims;
  params.padding_mode = padding_mode;
  params.interpolation_mode = interpolation_mode;
  params.align_corners = align_corners;

  for (const auto dim : c10::irange(dims)) {
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.grid_sizes[dim] = safe_downcast<int32_t, int64_t>(grid.size(dim));
    params.grid_strides[dim] = safe_downcast<int32_t, int64_t>(grid.stride(dim));
  }

  auto num_threads = output.numel();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc("grid_sampler_" + scalarToMetalTypeString(input));

      getMPSProfiler().beginProfileKernel(pso, op_name, {input, grid});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, output, input, grid, params);

      mtl_dispatch1DJob(computeEncoder, pso, num_threads);
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

} // namespace mps

Tensor grid_sampler_2d_mps(const Tensor& input,
                           const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode,
                           bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty({in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());

  mps::grid_sampler_2d_mps_impl(output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor grid_sampler_3d_mps(const Tensor& input,
                           const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode,
                           bool align_corners) {
  auto output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  mps::grid_sampler_template(output,
                             input,
                             grid,
                             interpolation_mode,
                             padding_mode,
                             align_corners,
                             /*sampler_dims=*/3,
                             /*op_name=*/"grid_sampler_3d");
  return output;
}

} // namespace at::native
