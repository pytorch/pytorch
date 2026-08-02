#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/zeros_like.h>
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

  const bool i32 = at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
      at::native::canUse32BitIndexMath(output);
  const auto idx_str = i32 ? "i32" : "i64";

  auto N = output.size(0);
  auto out_H = output.size(2);
  auto out_W = output.size(3);
  auto num_threads = N * out_H * out_W;

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("grid_sampler_2d_{}_{}_{}_{}",
                                                         interp_to_string(interpolation_mode),
                                                         padding_to_string(padding_mode),
                                                         idx_str,
                                                         scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(pso, "grid_sampler_2d", {input, grid});
      [computeEncoder setComputePipelineState:pso];
      auto dispatch = [&](auto idx_tag) {
        using IDX_T = decltype(idx_tag);
        GridSamplerParams<4, IDX_T> params(output, input, grid, align_corners);
        mtl_setArgs(computeEncoder, output, input, grid, params);
      };
      if (i32) {
        dispatch(int32_t{});
      } else {
        dispatch(int64_t{});
      }

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
    case GridSamplerInterpolation::Nearest:
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

  const bool i32 = at::native::canUse32BitIndexMath(input) && at::native::canUse32BitIndexMath(grid) &&
      at::native::canUse32BitIndexMath(output);
  const auto idx_str = i32 ? "i32" : "i64";

  auto num_threads = output.numel();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("grid_sampler_3d_{}_{}_{}_{}",
                                                         interp_to_string(interpolation_mode),
                                                         padding_to_string(padding_mode),
                                                         idx_str,
                                                         scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(pso, op_name, {input, grid});
      [computeEncoder setComputePipelineState:pso];
      auto dispatch = [&](auto idx_tag) {
        using IDX_T = decltype(idx_tag);
        GridSamplerParams<5, IDX_T> params(output, input, grid, align_corners);
        mtl_setArgs(computeEncoder, output, input, grid, params);
      };
      if (i32) {
        dispatch(int32_t{});
      } else {
        dispatch(int64_t{});
      }

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

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_mps(const Tensor& grad_output,
                                                        const Tensor& input,
                                                        const Tensor& grid,
                                                        int64_t _interpolation_mode,
                                                        int64_t _padding_mode,
                                                        bool align_corners,
                                                        std::array<bool, 2> output_mask) {
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "grid_sampler_2d_backward is not supported for complex on MPS");

  auto input_requires_grad = output_mask[0];
  auto grad_input = input_requires_grad ? at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : Tensor();
  auto interpolation_mode = static_cast<GridSamplerInterpolation>(_interpolation_mode);
  auto grad_grid = interpolation_mode == GridSamplerInterpolation::Nearest
      ? at::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto padding_mode = static_cast<GridSamplerPadding>(_padding_mode);

  auto N = input.size(0);
  auto out_H = grid.size(1);
  auto out_W = grid.size(2);
  auto num_threads = N * out_H * out_W;

  if (num_threads == 0) {
    return std::make_tuple(grad_input, grad_grid);
  }

  const bool i32 = at::native::canUse32BitIndexMath(grad_output) && at::native::canUse32BitIndexMath(input) &&
      at::native::canUse32BitIndexMath(grid) &&
      (!grad_input.defined() || at::native::canUse32BitIndexMath(grad_input)) &&
      at::native::canUse32BitIndexMath(grad_grid);
  const auto idx_str = i32 ? "i32" : "i64";

  using namespace mps;
  auto interp_str = mps::interp_to_string(interpolation_mode);
  auto pad_str = mps::padding_to_string(padding_mode);
  auto type_str = scalarToMetalTypeString(input);

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      auto set_args = [&](id<MTLComputeCommandEncoder> enc, auto&&... args) {
        auto with_idx = [&](auto idx_tag) {
          using IDX_T = decltype(idx_tag);
          GridSamplerBackwardParams<4, IDX_T> params(
              grad_output, input, grid, grad_input, grad_grid, align_corners, padding_mode, interpolation_mode);
          mtl_setArgs(enc, std::forward<decltype(args)>(args)..., params);
        };
        if (i32) {
          with_idx(int32_t{});
        } else {
          with_idx(int64_t{});
        }
      };

      if (input_requires_grad) {
        auto input_name = interpolation_mode == GridSamplerInterpolation::Bicubic
            ? fmt::format("grid_sampler_2d_backward_bicubic_input_{}_{}_{}", pad_str, idx_str, type_str)
            : fmt::format("grid_sampler_2d_backward_{}_input_{}_{}", interp_str, idx_str, type_str);
        auto input_pso = lib.getPipelineStateForFunc(input_name);
        getMPSProfiler().beginProfileKernel(input_pso, "grid_sampler_2d_backward_input", {grad_output, grid});
        [computeEncoder setComputePipelineState:input_pso];
        set_args(computeEncoder, grad_input, grad_output, grid);
        mtl_dispatch1DJob(computeEncoder, input_pso, num_threads);
        getMPSProfiler().endProfileKernel(input_pso);
      }

      if (interpolation_mode != GridSamplerInterpolation::Nearest) {
        auto grid_name = interpolation_mode == GridSamplerInterpolation::Bicubic
            ? fmt::format("grid_sampler_2d_backward_bicubic_grid_{}_{}_{}", pad_str, idx_str, type_str)
            : fmt::format("grid_sampler_2d_backward_bilinear_grid_{}_{}", idx_str, type_str);
        auto grid_pso = lib.getPipelineStateForFunc(grid_name);
        getMPSProfiler().beginProfileKernel(grid_pso, "grid_sampler_2d_backward_grid", {grad_output, input, grid});
        [computeEncoder setComputePipelineState:grid_pso];
        set_args(computeEncoder, grad_grid, grad_output, input, grid);
        mtl_dispatch1DJob(computeEncoder, grid_pso, num_threads);
        getMPSProfiler().endProfileKernel(grid_pso);
      }
    }
  });

  return std::make_tuple(std::move(grad_input), std::move(grad_grid));
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_mps(const Tensor& grad_output,
                                                        const Tensor& input,
                                                        const Tensor& grid,
                                                        int64_t interpolation_mode,
                                                        int64_t padding_mode,
                                                        bool align_corners,
                                                        std::array<bool, 2> output_mask) {
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  TORCH_CHECK_NOT_IMPLEMENTED(interpolation_mode == 0 || interpolation_mode == 1,
                              "grid_sampler_3d backward on MPS only supports bilinear and nearest interpolation");

  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  auto input_requires_grad = output_mask[0];
  auto interp_mode = static_cast<GridSamplerInterpolation>(interpolation_mode);
  auto pad_mode = static_cast<GridSamplerPadding>(padding_mode);

  Tensor grad_input;
  if (input_requires_grad) {
    grad_input = at::zeros_like(input);
  }
  // Always allocate grad_grid, matching CPU/CUDA and the 2D MPS backward.
  // Autograd requires a defined tensor for every output declared in the
  // derivative, even when the corresponding input doesn't require grad.
  auto grad_grid = interp_mode == GridSamplerInterpolation::Nearest ? at::zeros_like(grid, MemoryFormat::Contiguous)
                                                                    : at::empty_like(grid, MemoryFormat::Contiguous);

  const auto& input_contiguous = input.contiguous();
  const auto& grid_contiguous = grid.contiguous();
  const auto& grad_output_contiguous = grad_output.contiguous();

  auto N = input_contiguous.size(0);
  auto out_D = grid_contiguous.size(1);
  auto out_H = grid_contiguous.size(2);
  auto out_W = grid_contiguous.size(3);

  bool run_grad_input = input_requires_grad;
  bool run_grad_grid = interp_mode != GridSamplerInterpolation::Nearest;

  if (!run_grad_input && !run_grad_grid) {
    return std::make_tuple(std::move(grad_input), std::move(grad_grid));
  }

  // The combined kernel needs a valid buffer pointer for grad_input even when
  // it is not requested, so allocate a dummy with the expected rank so stride
  // queries below remain in range.
  auto grad_input_buf = run_grad_input ? grad_input : at::zeros({1, 1, 1, 1, 1}, input.options());

  const bool i32 = at::native::canUse32BitIndexMath(grad_output_contiguous) &&
      at::native::canUse32BitIndexMath(input_contiguous) && at::native::canUse32BitIndexMath(grid_contiguous) &&
      at::native::canUse32BitIndexMath(grad_input_buf) && at::native::canUse32BitIndexMath(grad_grid);
  const auto idx_str = i32 ? "i32" : "i64";

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      auto pso = lib.getPipelineStateForFunc(
          fmt::format("grid_sampler_3d_backward_{}_{}", idx_str, scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(
          pso,
          "grid_sampler_3d_backward",
          {grad_output_contiguous, input_contiguous, grid_contiguous, grad_input_buf, grad_grid});

      [computeEncoder setComputePipelineState:pso];

      auto dispatch = [&](auto idx_tag) {
        using IDX_T = decltype(idx_tag);
        GridSamplerBackwardParams<5, IDX_T> params(grad_output_contiguous,
                                                   input_contiguous,
                                                   grid_contiguous,
                                                   grad_input,
                                                   grad_grid,
                                                   align_corners,
                                                   pad_mode,
                                                   interp_mode);
        mtl_setArgs(computeEncoder,
                    grad_output_contiguous,
                    input_contiguous,
                    grid_contiguous,
                    grad_input_buf,
                    grad_grid,
                    params);
      };
      if (i32) {
        dispatch(int32_t{});
      } else {
        dispatch(int64_t{});
      }

      MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
      MTLSize threadsPerGrid = MTLSizeMake(out_W, out_H * out_D, N);
      [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return std::make_tuple(std::move(grad_input), std::move(grad_grid));
}

} // namespace at::native
