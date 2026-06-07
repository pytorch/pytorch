#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SegmentReduce_metallib.h>
#endif

static Tensor _segment_reduce_lengths_mps_kernel(ReductionType reduction,
                                                 const Tensor& data,
                                                 const Tensor& lengths,
                                                 int64_t axis,
                                                 const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_lengths_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());

  auto outer_shape = data.sizes().vec();
  int64_t segment_count = lengths.size(-1);
  outer_shape[axis] = segment_count;
  auto output = at::empty(outer_shape, data.options());

  if (output.numel() == 0) {
    return output;
  }

  int64_t outer_offset = 1;
  for (int i = 0; i < axis; i++) {
    outer_offset *= data.size(i);
  }

  int64_t inner_offset = 1;
  for (int i = axis + 1; i < data.dim(); i++) {
    inner_offset *= data.size(i);
  }

  int64_t data_size_axis = data.size(axis);

  auto lengths_2d = lengths.reshape({outer_offset, segment_count}).to(at::kInt);

  auto zeros = at::zeros({outer_offset, 1}, lengths_2d.options());
  auto lengths_cumsum = at::cat({zeros, lengths_2d}, 1).cumsum(1).to(at::kInt).contiguous();

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_forward_" + scalarToMetalTypeString(data);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_mps", [&] {
        scalar_t initial_val = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(0);
        mtl_setArgs(computeEncoder,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    segment_count,
                    outer_offset,
                    inner_offset,
                    data_size_axis,
                    static_cast<int32_t>(reduction),
                    initial_val,
                    initial.has_value());
        int64_t total_threads = outer_offset * segment_count * inner_offset;
        int64_t maxThreadgroups = 1024;
        int64_t maxThreads = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tgSize = std::min(maxThreads, total_threads);
        MTLSize threadgroupsPerGrid =
            MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(total_threads, tgSize)), 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
        [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
      });
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return output;
}

static Tensor _segment_reduce_offsets_mps_kernel(ReductionType reduction,
                                                 const Tensor& data,
                                                 const Tensor& offsets,
                                                 int64_t axis,
                                                 const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_offsets_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());

  auto outer_shape = data.sizes().vec();
  // offsets has shape [..., segment_count + 1]; subtract 1 to recover segment_count.
  int64_t segment_count = offsets.size(-1) - 1;
  outer_shape[axis] = segment_count;
  auto output = at::empty(outer_shape, data.options());

  if (output.numel() == 0) {
    return output;
  }

  int64_t outer_offset = 1;
  for (int i = 0; i < axis; i++) {
    outer_offset *= data.size(i);
  }

  int64_t inner_offset = 1;
  for (int i = axis + 1; i < data.dim(); i++) {
    inner_offset *= data.size(i);
  }

  int64_t data_size_axis = data.size(axis);

  // offsets is already the cumulative sum: reshape + cast to int32 and we are done.
  auto lengths_cumsum = offsets.reshape({outer_offset, segment_count + 1}).to(at::kInt).contiguous();
  // lengths_data buffer is not actually read by the kernel; pass cumsum as a placeholder.
  auto lengths_2d = lengths_cumsum;

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_forward_" + scalarToMetalTypeString(data);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_mps", [&] {
        scalar_t initial_val = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(0);
        mtl_setArgs(computeEncoder,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    segment_count,
                    outer_offset,
                    inner_offset,
                    data_size_axis,
                    static_cast<int32_t>(reduction),
                    initial_val,
                    initial.has_value());
        int64_t total_threads = outer_offset * segment_count * inner_offset;
        int64_t maxThreadgroups = 1024;
        int64_t maxThreads = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tgSize = std::min(maxThreads, total_threads);
        MTLSize threadgroupsPerGrid =
            MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(total_threads, tgSize)), 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
        [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
      });
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return output;
}

static Tensor _segment_reduce_lengths_backward_mps_kernel(const Tensor& grad,
                                                          const Tensor& output,
                                                          const Tensor& data,
                                                          ReductionType reduction,
                                                          const Tensor& lengths,
                                                          int64_t axis,
                                                          const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_lengths_backward_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());

  auto grad_input = at::zeros(data.sizes(), grad.options());

  if (grad_input.numel() == 0) {
    return grad_input;
  }

  int64_t segment_count = lengths.size(-1);

  int64_t outer_offset = 1;
  for (int i = 0; i < axis; i++) {
    outer_offset *= data.size(i);
  }

  int64_t inner_offset = 1;
  for (int i = axis + 1; i < data.dim(); i++) {
    inner_offset *= data.size(i);
  }

  int64_t data_size_axis = data.size(axis);

  auto lengths_2d = lengths.reshape({outer_offset, segment_count}).to(at::kInt);
  auto cumsum_zeros = at::zeros({outer_offset, 1}, lengths_2d.options());
  auto lengths_cumsum = at::cat({cumsum_zeros, lengths_2d}, 1).cumsum(1).to(at::kInt).contiguous();

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_backward_" + scalarToMetalTypeString(data);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_backward_mps", [&] {
        scalar_t initial_prod_value = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(1);
        mtl_setArgs(computeEncoder,
                    grad_input,
                    grad,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    segment_count,
                    outer_offset,
                    inner_offset,
                    data_size_axis,
                    static_cast<int32_t>(reduction),
                    initial_prod_value);
        int64_t total_threads = outer_offset * segment_count * inner_offset;
        int64_t maxThreadgroups = 1024;
        int64_t maxThreads = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tgSize = std::min(maxThreads, total_threads);
        MTLSize threadgroupsPerGrid =
            MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(total_threads, tgSize)), 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
        [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
      });
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return grad_input;
}

static Tensor _segment_reduce_offsets_backward_mps_kernel(const Tensor& grad,
                                                          const Tensor& output,
                                                          const Tensor& data,
                                                          ReductionType reduction,
                                                          const Tensor& offsets,
                                                          int64_t axis,
                                                          const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_offsets_backward_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());

  auto grad_input = at::zeros(data.sizes(), grad.options());

  if (grad_input.numel() == 0) {
    return grad_input;
  }

  int64_t segment_count = offsets.size(-1) - 1;

  int64_t outer_offset = 1;
  for (int i = 0; i < axis; i++) {
    outer_offset *= data.size(i);
  }

  int64_t inner_offset = 1;
  for (int i = axis + 1; i < data.dim(); i++) {
    inner_offset *= data.size(i);
  }

  int64_t data_size_axis = data.size(axis);

  // offsets is already the cumulative sum.
  auto lengths_cumsum = offsets.reshape({outer_offset, segment_count + 1}).to(at::kInt).contiguous();
  auto lengths_2d = lengths_cumsum;

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_backward_" + scalarToMetalTypeString(data);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_backward_mps", [&] {
        scalar_t initial_prod_value = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(1);
        mtl_setArgs(computeEncoder,
                    grad_input,
                    grad,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    segment_count,
                    outer_offset,
                    inner_offset,
                    data_size_axis,
                    static_cast<int32_t>(reduction),
                    initial_prod_value);
        int64_t total_threads = outer_offset * segment_count * inner_offset;
        int64_t maxThreadgroups = 1024;
        int64_t maxThreads = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tgSize = std::min(maxThreads, total_threads);
        MTLSize threadgroupsPerGrid =
            MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(total_threads, tgSize)), 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
        [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];
      });
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return grad_input;
}

} // namespace mps

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &mps::_segment_reduce_lengths_mps_kernel);
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &mps::_segment_reduce_lengths_backward_mps_kernel);

REGISTER_DISPATCH(_segment_reduce_offsets_stub, &mps::_segment_reduce_offsets_mps_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &mps::_segment_reduce_offsets_backward_mps_kernel);

} // namespace at::native
