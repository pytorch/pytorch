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

static const char* interp_to_string(GridSamplerInterpolation mode) {
  switch (mode) {
    case GridSamplerInterpolation::Bilinear:
      return "bilinear";
    case GridSamplerInterpolation::Nearest:
      return "nearest";
    case GridSamplerInterpolation::Bicubic:
      return "bicubic";
  }
  TORCH_CHECK(false, "Unrecognised interpolation mode: ", mode);
  return "";
}

static const char* padding_to_string(GridSamplerPadding mode) {
  switch (mode) {
    case GridSamplerPadding::Zeros:
      return "zeros";
    case GridSamplerPadding::Border:
      return "border";
    case GridSamplerPadding::Reflection:
      return "reflection";
  }
  TORCH_CHECK(false, "Unrecognised padding mode: ", mode);
  return "";
}

static void grid_sampler_2d_mps_impl(Tensor& output,
                                     const Tensor& input,
                                     const Tensor& grid,
                                     int64_t _interpolation_mode,
                                     int64_t _padding_mode,
                                     bool align_corners) {
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "grid_sampler_2d is not supported for complex on MPS");

  auto interpolation_mode = static_cast<GridSamplerInterpolation>(_interpolation_mode);
  auto padding_mode = static_cast<GridSamplerPadding>(_padding_mode);

  auto dims = input.dim();

  GridSamplerParams<4> params;
  params.sampler_dims = 2;
  params.align_corners = align_corners;

  for (const auto dim : c10::irange(dims)) {
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.grid_sizes[dim] = safe_downcast<int32_t, int64_t>(grid.size(dim));
    params.grid_strides[dim] = safe_downcast<int32_t, int64_t>(grid.stride(dim));
  }

  auto N = output.size(0);
  auto out_H = output.size(2);
  auto out_W = output.size(3);
  auto num_threads = N * out_H * out_W;

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("grid_sampler_2d_{}_{}_{}",
                                                         interp_to_string(interpolation_mode),
                                                         padding_to_string(padding_mode),
                                                         scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(pso, "grid_sampler_2d", {input, grid});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, output, input, grid, params);

      mtl_dispatch1DJob(computeEncoder, pso, num_threads);
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

static void grid_sampler_3d_mps_impl(Tensor& output,
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

  auto input_size = input.sizes();
  auto grid_size = grid.sizes();
  output.resize_({input_size[0], input_size[1], grid_size[1], grid_size[2], grid_size[3]}, MemoryFormat::Contiguous);

  auto dims = input.dim();

  GridSamplerParams<5> params;
  params.sampler_dims = sampler_dims;
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
      auto pso = lib.getPipelineStateForFunc(
          fmt::format("grid_sampler_3d_{}_{}", padding_to_string(padding_mode), scalarToMetalTypeString(input)));

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
  mps::grid_sampler_3d_mps_impl(output,
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
