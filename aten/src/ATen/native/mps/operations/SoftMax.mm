//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SoftMaxKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SoftMaxKernel_metallib.h>
#endif

static bool canUseMetalSoftmax(const Tensor& input, int64_t dim) {
  return input.dim() > 0;
}

static SoftmaxParams makeForwardParams(const Tensor& input, const Tensor& output, int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = input.dim();
  params.axis_size = static_cast<uint32_t>(input.size(dim));
  params.stride_a = static_cast<uint32_t>(input.stride(dim));
  params.stride_b = static_cast<uint32_t>(output.stride(dim));
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(input.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(input.stride(d));
    params.outer_strides_b[outer_idx] = static_cast<uint32_t>(output.stride(d));
    outer_idx++;
  }
  return params;
}

static SoftmaxParams makeBackwardParams(const Tensor& grad, const Tensor& output, const Tensor& grad_input, int64_t dim) {
  SoftmaxParams params = {};
  int64_t ndim = grad.dim();
  params.axis_size = static_cast<uint32_t>(grad.size(dim));
  params.stride_a = static_cast<uint32_t>(grad.stride(dim));
  params.stride_b = static_cast<uint32_t>(output.stride(dim));
  params.stride_c = static_cast<uint32_t>(grad_input.stride(dim));
  params.ndim = static_cast<uint32_t>(ndim);
  int outer_idx = 0;
  for (int64_t d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    params.outer_sizes[outer_idx] = static_cast<uint32_t>(grad.size(d));
    params.outer_strides_a[outer_idx] = static_cast<uint32_t>(grad.stride(d));
    params.outer_strides_b[outer_idx] = static_cast<uint32_t>(output.stride(d));
    params.outer_strides_c[outer_idx] = static_cast<uint32_t>(grad_input.stride(d));
    outer_idx++;
  }
  return params;
}

} // namespace mps

TORCH_IMPL_FUNC(softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "softmax only supported for floating types");

  if (input_.numel() == 0) {
    return;
  }

  Tensor input;
  if (input_.dim() == 0) {
    input = input_.view(1);
  } else
    input = input_;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "Softmax:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(input, dim_)) {
    using namespace mps;
    int64_t axis_size = input.size(dim_);
    int64_t outer_size = input.numel() / axis_size;
    auto params = makeForwardParams(input, output, dim_);

    // Tiled path: each thread does one complete softmax row at one inner position.
    // Gives perfect memory coalescing for non-last-dim with large inner_size.
    {
      int64_t ndim = input.dim();
      bool use_tiled = (dim_ != ndim - 1) && input.is_contiguous() && output.is_contiguous();
      int64_t inner_size = input.stride(dim_);
      use_tiled = use_tiled && (inner_size >= axis_size);
      if (use_tiled) {
        int64_t outer_before = outer_size / inner_size;
        int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
        int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
          tile_tg_size /= 2;
          num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        }
        params.num_chunks = static_cast<uint32_t>(num_tiles);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(input);
            auto kernel = mps::lib.getPipelineStateForFunc("softmax_forward_tiled_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, input, output, params);
            MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    // Coalesced path: flat loads with shared memory reduction
    {
      int64_t ndim = input.dim();
      int64_t inner_size = input.stride(dim_);
      bool use_coalesced = (dim_ != ndim - 1) && input.is_contiguous() && output.is_contiguous() && (inner_size > 1) && (inner_size < axis_size) && (axis_size <= 16384);
      if (use_coalesced) {
        int64_t outer_before = outer_size / inner_size;
        int64_t nat = 1;
        while (nat * 2 <= 1024 / inner_size) nat *= 2;
        int64_t coal_tg_size = inner_size * nat;
        params.num_chunks = static_cast<uint32_t>(nat);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(input);
            auto kernel = mps::lib.getPipelineStateForFunc("softmax_forward_coalesced_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, input, output, params);
            [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
            MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    constexpr int N_READS = 4;
    int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));

    constexpr int64_t kFwdMinOccupancyTG = 8;
    int64_t elems_per_tg = tg_size * N_READS;
    int64_t raw_chunks = axis_size / elems_per_tg;
    int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
    bool use_two_pass_fwd = (raw_chunks >= 8) && (outer_size < kFwdMinOccupancyTG);

    Tensor fwd_partials;
    if (use_two_pass_fwd) {
      params.num_chunks = static_cast<uint32_t>(max_chunks);
      fwd_partials = at::empty({outer_size * max_chunks * 2}, input.options().dtype(at::kFloat));
    }

    MPSStream* stream = getCurrentMPSStream();

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        auto metalType = mps::scalarToMetalTypeString(input);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

        if (use_two_pass_fwd) {
          auto reduce_kernel = mps::lib.getPipelineStateForFunc("softmax_forward_2pass_reduce_" + metalType);
          [encoder setComputePipelineState:reduce_kernel];
          mps::mtl_setArgs(encoder, input, fwd_partials, params);
          MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          auto write_kernel = mps::lib.getPipelineStateForFunc("softmax_forward_2pass_write_" + metalType);
          [encoder setComputePipelineState:write_kernel];
          mps::mtl_setArgs(encoder, input, output, fwd_partials, params);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        } else {
          id<MTLComputePipelineState> kernel;
          if (axis_size <= 1024 * N_READS) {
            kernel = mps::lib.getPipelineStateForFunc("softmax_forward_single_row_" + metalType);
          } else {
            kernel = mps::lib.getPipelineStateForFunc("softmax_forward_looped_" + metalType);
          }

          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, input, output, params);
          MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }
      });
    }

    return;
  }
}

TORCH_IMPL_FUNC(softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(output, dim_) && mps::canUseMetalSoftmax(grad, dim_)) {
    using namespace mps;
    int64_t axis_size = output.size(dim_);
    int64_t outer_size = output.numel() / axis_size;

    constexpr int N_READS = 4;
    int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));
    auto params = makeBackwardParams(grad, output, grad_input, dim_);

    // Tiled path for non-last-dim backward
    {
      int64_t ndim = grad.dim();
      bool use_tiled = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() && grad_input.is_contiguous();
      int64_t inner_size = grad.stride(dim_);
      use_tiled = use_tiled && (inner_size >= axis_size);
      if (use_tiled) {
        int64_t outer_before = outer_size / inner_size;
        int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
        int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
          tile_tg_size /= 2;
          num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        }
        params.num_chunks = static_cast<uint32_t>(num_tiles);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(output);
            auto kernel = mps::lib.getPipelineStateForFunc("softmax_backward_tiled_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, grad, output, grad_input, params);
            MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    // Coalesced path: flat loads with shared memory reduction
    {
      int64_t ndim = grad.dim();
      int64_t inner_size = grad.stride(dim_);
      bool use_coalesced = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() && grad_input.is_contiguous() && (inner_size > 1) && (inner_size < axis_size) && (axis_size <= 16384);
      if (use_coalesced) {
        int64_t outer_before = outer_size / inner_size;
        int64_t nat = 1;
        while (nat * 2 <= 1024 / inner_size) nat *= 2;
        int64_t coal_tg_size = inner_size * nat;
        params.num_chunks = static_cast<uint32_t>(nat);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(output);
            auto kernel = mps::lib.getPipelineStateForFunc("softmax_backward_coalesced_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, grad, output, grad_input, params);
            [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
            MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    constexpr int64_t kMinOccupancyTG = 8;
    int64_t elems_per_tg = tg_size * N_READS;
    int64_t raw_chunks = axis_size / elems_per_tg;
    int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
    bool use_two_pass = (raw_chunks >= 8) && (outer_size < kMinOccupancyTG);

    Tensor partial_sums;
    if (use_two_pass) {
      params.num_chunks = static_cast<uint32_t>(max_chunks);
      partial_sums = at::empty({outer_size * max_chunks}, grad.options().dtype(at::kFloat));
    }

    MPSStream* stream = getCurrentMPSStream();

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        auto metalType = mps::scalarToMetalTypeString(output);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

        if (use_two_pass) {
          auto dot_kernel = mps::lib.getPipelineStateForFunc("softmax_backward_2pass_dot_" + metalType);
          [encoder setComputePipelineState:dot_kernel];
          mps::mtl_setArgs(encoder, grad, output, partial_sums, params);
          MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          auto grad_kernel = mps::lib.getPipelineStateForFunc("softmax_backward_2pass_grad_" + metalType);
          [encoder setComputePipelineState:grad_kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, partial_sums, params);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        } else {
          id<MTLComputePipelineState> kernel;
          if (axis_size <= 1024 * N_READS) {
            kernel = mps::lib.getPipelineStateForFunc("softmax_backward_single_row_" + metalType);
          } else {
            kernel = mps::lib.getPipelineStateForFunc("softmax_backward_looped_" + metalType);
          }

          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, params);
          MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }
      });
    }

    return;
  }
}


TORCH_IMPL_FUNC(log_softmax_mps_out)
(const Tensor& input_, const int64_t dim, const bool half_to_float, const Tensor& output) {
  TORCH_CHECK(!half_to_float, "log_softmax with half to float conversion is not supported on MPS");
  TORCH_CHECK(c10::isFloatingType(input_.scalar_type()), "log_softmax only supported for floating types");

  if (input_.numel() == 0) {
    return;
  }

  Tensor input;
  if (input_.dim() == 0) {
    input = input_.view(1);
  } else
    input = input_;

  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(), "LogSoftmax:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(input, dim_)) {
    using namespace mps;
    int64_t axis_size = input.size(dim_);
    int64_t outer_size = input.numel() / axis_size;
    auto params = makeForwardParams(input, output, dim_);

    // Tiled path for non-last-dim log_softmax forward
    {
      int64_t ndim = input.dim();
      bool use_tiled = (dim_ != ndim - 1) && input.is_contiguous() && output.is_contiguous();
      int64_t inner_size = input.stride(dim_);
      use_tiled = use_tiled && (inner_size >= axis_size);
      if (use_tiled) {
        int64_t outer_before = outer_size / inner_size;
        int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
        int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
          tile_tg_size /= 2;
          num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        }
        params.num_chunks = static_cast<uint32_t>(num_tiles);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(input);
            auto kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_tiled_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, input, output, params);
            MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    // Coalesced path: flat loads with shared memory reduction
    {
      int64_t ndim = input.dim();
      int64_t inner_size = input.stride(dim_);
      bool use_coalesced = (dim_ != ndim - 1) && input.is_contiguous() && output.is_contiguous() && (inner_size > 1) && (inner_size < axis_size) && (axis_size <= 16384);
      if (use_coalesced) {
        int64_t outer_before = outer_size / inner_size;
        int64_t nat = 1;
        while (nat * 2 <= 1024 / inner_size) nat *= 2;
        int64_t coal_tg_size = inner_size * nat;
        params.num_chunks = static_cast<uint32_t>(nat);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(input);
            auto kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_coalesced_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, input, output, params);
            [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
            MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    MPSStream* stream = getCurrentMPSStream();

    constexpr int N_READS = 4;
    int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));
    int64_t elems_per_tg = tg_size * N_READS;

    constexpr int64_t kMinOccupancyTG = 8;
    int64_t raw_chunks = axis_size / elems_per_tg;
    int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
    bool use_two_pass = (raw_chunks >= 8) && (outer_size < kMinOccupancyTG);

    Tensor partials;
    if (use_two_pass) {
      params.num_chunks = static_cast<uint32_t>(max_chunks);
      partials = at::empty({outer_size * max_chunks * 2}, input.options().dtype(at::kFloat));
    }

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        auto metalType = mps::scalarToMetalTypeString(input);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

        if (use_two_pass) {
          auto reduce_kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_2pass_reduce_" + metalType);
          [encoder setComputePipelineState:reduce_kernel];
          mps::mtl_setArgs(encoder, input, partials, params);
          MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          auto write_kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_2pass_write_" + metalType);
          [encoder setComputePipelineState:write_kernel];
          mps::mtl_setArgs(encoder, input, output, partials, params);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        } else {
          id<MTLComputePipelineState> kernel;
          if (axis_size <= 1024 * N_READS) {
            kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_single_row_" + metalType);
          } else {
            kernel = mps::lib.getPipelineStateForFunc("log_softmax_forward_looped_" + metalType);
          }

          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, input, output, params);
          MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }
      });
    }

    return;
  }
}

TORCH_IMPL_FUNC(log_softmax_backward_mps_out)
(const Tensor& grad_, const Tensor& output_, int64_t dim, ScalarType input_dtype, const Tensor& grad_input) {
  if (output_.numel() == 0) {
    return;
  }

  Tensor grad;
  if (grad_.dim() == 0) {
    grad = grad_.view(1);
  } else
    grad = grad_;

  Tensor output;
  if (output_.dim() == 0) {
    output = output_.view(1);
  } else
    output = output_;

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TORCH_CHECK(dim_ >= 0 && dim_ < grad.dim(), "Grad:dim must be non-negative and less than input dimensions");

  if (mps::canUseMetalSoftmax(output, dim_) && mps::canUseMetalSoftmax(grad, dim_)) {
    using namespace mps;
    int64_t axis_size = output.size(dim_);
    int64_t outer_size = output.numel() / axis_size;

    constexpr int N_READS = 4;
    int64_t tg_size = std::min(static_cast<int64_t>((axis_size + N_READS - 1) / N_READS), static_cast<int64_t>(1024));
    auto params = makeBackwardParams(grad, output, grad_input, dim_);

    // Tiled path for non-last-dim log_softmax backward
    {
      int64_t ndim = grad.dim();
      bool use_tiled = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() && grad_input.is_contiguous();
      int64_t inner_size = grad.stride(dim_);
      use_tiled = use_tiled && (inner_size >= axis_size);
      if (use_tiled) {
        int64_t outer_before = outer_size / inner_size;
        int64_t tile_tg_size = std::min(inner_size, static_cast<int64_t>(1024));
        int64_t num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        while (num_tiles * outer_before < 64 && tile_tg_size > 32) {
          tile_tg_size /= 2;
          num_tiles = (inner_size + tile_tg_size - 1) / tile_tg_size;
        }
        params.num_chunks = static_cast<uint32_t>(num_tiles);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(output);
            auto kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_tiled_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, grad, output, grad_input, params);
            MTLSize threadsPerGroup = MTLSizeMake(tile_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(num_tiles * outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    // Coalesced path: flat loads with shared memory reduction
    {
      int64_t ndim = grad.dim();
      int64_t inner_size = grad.stride(dim_);
      bool use_coalesced = (dim_ != ndim - 1) && grad.is_contiguous() && output.is_contiguous() && grad_input.is_contiguous() && (inner_size > 1) && (inner_size < axis_size) && (axis_size <= 16384);
      if (use_coalesced) {
        int64_t outer_before = outer_size / inner_size;
        int64_t nat = 1;
        while (nat * 2 <= 1024 / inner_size) nat *= 2;
        int64_t coal_tg_size = inner_size * nat;
        params.num_chunks = static_cast<uint32_t>(nat);

        MPSStream* stream = getCurrentMPSStream();
        @autoreleasepool {
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            auto metalType = mps::scalarToMetalTypeString(output);
            auto kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_coalesced_" + metalType);
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernel];
            mps::mtl_setArgs(encoder, grad, output, grad_input, params);
            [encoder setThreadgroupMemoryLength:coal_tg_size * 2 * sizeof(float) atIndex:0];
            MTLSize threadsPerGroup = MTLSizeMake(coal_tg_size, 1, 1);
            MTLSize numGroups = MTLSizeMake(outer_before, 1, 1);
            [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
          });
        }
        return;
      }
    }

    constexpr int64_t kMinOccupancyTG = 8;
    int64_t elems_per_tg = tg_size * N_READS;
    int64_t raw_chunks = axis_size / elems_per_tg;
    int64_t max_chunks = std::min(raw_chunks, static_cast<int64_t>(16));
    bool use_two_pass = (raw_chunks >= 8) && (outer_size < kMinOccupancyTG);

    Tensor partial_sums;
    if (use_two_pass) {
      params.num_chunks = static_cast<uint32_t>(max_chunks);
      partial_sums = at::empty({outer_size * max_chunks}, grad.options().dtype(at::kFloat));
    }

    MPSStream* stream = getCurrentMPSStream();

    @autoreleasepool {
      dispatch_sync_with_rethrow(stream->queue(), ^() {
        auto metalType = mps::scalarToMetalTypeString(output);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        MTLSize threadsPerGroup = MTLSizeMake(tg_size, 1, 1);

        if (use_two_pass) {
          auto sum_kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_2pass_sum_" + metalType);
          [encoder setComputePipelineState:sum_kernel];
          mps::mtl_setArgs(encoder, grad, partial_sums, params);
          MTLSize numGroups = MTLSizeMake(static_cast<NSUInteger>(params.num_chunks) * outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];

          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

          auto grad_kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_2pass_grad_" + metalType);
          [encoder setComputePipelineState:grad_kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, partial_sums, params);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        } else {
          id<MTLComputePipelineState> kernel;
          if (axis_size <= 1024 * N_READS) {
            kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_single_row_" + metalType);
          } else {
            kernel = mps::lib.getPipelineStateForFunc("log_softmax_backward_looped_" + metalType);
          }

          [encoder setComputePipelineState:kernel];
          mps::mtl_setArgs(encoder, grad, output, grad_input, params);
          MTLSize numGroups = MTLSizeMake(outer_size, 1, 1);
          [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
        }
      });
    }

    return;
  }
}

} // namespace at::native
