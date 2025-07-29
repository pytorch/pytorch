#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSProfiler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GridSampler3D_metallib.h>
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

  MPSGraphResizeMode samplingMode = MPSGraphResizeBilinear;
  MPSGraphPaddingMode paddingMode = MPSGraphPaddingModeZero;

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

static void grid_sampler_3d_mps_impl(Tensor& output,
                                     const Tensor& input,
                                     const Tensor& grid,
                                     int64_t interpolation_mode,
                                     int64_t padding_mode,
                                     bool align_corners) {
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();

  // Kernel selection logic
  bool use_vectorized = false;

  // Environment variable override (useful for development/benchmarking)
  const char* env_kernel = std::getenv("PYTORCH_MPS_GRID_SAMPLER_3D_KERNEL");
  if (env_kernel) {
    if (std::string(env_kernel) == "vectorized") {
      use_vectorized = true;
    }
  } else {
    // Adaptive kernel selection based on tensor size and characteristics
    int64_t output_elements = input.size(0) * input.size(1) * grid.size(1) * grid.size(2) * grid.size(3);

    // Use vectorized kernel for larger tensors (> 1M elements) or when width is large enough
    use_vectorized = (output_elements > 1000000) || (grid.size(3) >= 64);
  }

  std::string kernel_name;
  if (use_vectorized) {
    kernel_name = "grid_sampler_3d_vectorized_" + mps::scalarToMetalTypeString(input);
  } else {
    kernel_name = "grid_sampler_3d_" + mps::scalarToMetalTypeString(input);
  }

  auto gridSampler3DPSO = lib.getPipelineStateForFunc(kernel_name);

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      // Prepare input and output tensors
      auto input_contiguous = input.contiguous();
      auto grid_contiguous = grid.contiguous();

      // Get tensor dimensions
      int64_t N = input_contiguous.size(0);
      int64_t C = input_contiguous.size(1);
      int64_t in_D = input_contiguous.size(2);
      int64_t in_H = input_contiguous.size(3);
      int64_t in_W = input_contiguous.size(4);

      int64_t out_D = grid_contiguous.size(1);
      int64_t out_H = grid_contiguous.size(2);
      int64_t out_W = grid_contiguous.size(3);

            // Prepare kernel arguments - all 5 dimensions and strides
      std::array<uint64_t, 5> input_sizes = {
        static_cast<uint64_t>(N),
        static_cast<uint64_t>(C),
        static_cast<uint64_t>(in_D),
        static_cast<uint64_t>(in_H),
        static_cast<uint64_t>(in_W)
      };
      std::array<uint64_t, 5> output_sizes = {
        static_cast<uint64_t>(N),
        static_cast<uint64_t>(C),
        static_cast<uint64_t>(out_D),
        static_cast<uint64_t>(out_H),
        static_cast<uint64_t>(out_W)
      };
      std::array<uint64_t, 5> input_strides = {
        static_cast<uint64_t>(input_contiguous.stride(0)),
        static_cast<uint64_t>(input_contiguous.stride(1)),
        static_cast<uint64_t>(input_contiguous.stride(2)),
        static_cast<uint64_t>(input_contiguous.stride(3)),
        static_cast<uint64_t>(input_contiguous.stride(4))
      };
      std::array<uint64_t, 5> output_strides = {
        static_cast<uint64_t>(output.stride(0)),
        static_cast<uint64_t>(output.stride(1)),
        static_cast<uint64_t>(output.stride(2)),
        static_cast<uint64_t>(output.stride(3)),
        static_cast<uint64_t>(output.stride(4))
      };
      std::array<uint64_t, 5> grid_strides = {
        static_cast<uint64_t>(grid_contiguous.stride(0)),
        static_cast<uint64_t>(grid_contiguous.stride(1)),
        static_cast<uint64_t>(grid_contiguous.stride(2)),
        static_cast<uint64_t>(grid_contiguous.stride(3)),
        static_cast<uint64_t>(grid_contiguous.stride(4))  // coordinate stride
      };

      int32_t interp_mode = static_cast<int32_t>(interpolation_mode);
      int32_t pad_mode = static_cast<int32_t>(padding_mode);

      getMPSProfiler().beginProfileKernel(gridSampler3DPSO, "grid_sampler_3d", {input_contiguous, grid_contiguous, output});

      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:gridSampler3DPSO];

                  mtl_setArgs(computeEncoder,
                  input_contiguous,
                  output,
                  grid_contiguous,
                  interp_mode,
                  pad_mode,
                  align_corners,
                  input_sizes,
                  output_sizes,
                  input_strides,
                  output_strides,
                  grid_strides);

      const uint32_t TILE_SIZE = 16;
      MTLSize threadsPerThreadgroup = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);

      MTLSize threadsPerGrid;
      if (use_vectorized) {
        // For vectorized kernel: each thread processes 4 elements in width dimension
        const uint32_t ELEMS_PER_THREAD = 4;
        threadsPerGrid = MTLSizeMake((out_W + ELEMS_PER_THREAD - 1) / ELEMS_PER_THREAD, out_H * out_D, N * C);
      } else {
        // For standard kernel: one thread per output element
        threadsPerGrid = MTLSizeMake(out_W, out_H * out_D, N * C);
      }

      [computeEncoder dispatchThreads:threadsPerGrid
                threadsPerThreadgroup:threadsPerThreadgroup];

      getMPSProfiler().endProfileKernel(gridSampler3DPSO);
    }
  });
}

} // namespace mps

Tensor grid_sampler_2d_mps(const Tensor& input,
                           const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode,
                           bool align_corners) {
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS)) {
    TORCH_WARN_ONCE("MPS: grid_sampler_2d op is supported natively starting from macOS 13.2. ",
                    "Falling back on CPU. This may have performance implications.");

    return at::grid_sampler_2d(input.to("cpu"), grid.to("cpu"), interpolation_mode, padding_mode, align_corners)
        .clone()
        .to("mps");
  }

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
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty({in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]}, input.options());

  mps::grid_sampler_3d_mps_impl(output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

} // namespace at::native
