#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SegmentReduce.h>

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

static std::string reductionToMetalString(ReductionType reduction) {
  switch (reduction) {
    case ReductionType::MAX:  return "MAX";
    case ReductionType::MEAN: return "MEAN";
    case ReductionType::MIN:  return "MIN";
    case ReductionType::SUM:  return "SUM";
    case ReductionType::PROD: return "PROD";
  }
}

static inline at::ScalarType mps_index_dtype(const Tensor& t) {
  return at::native::canUse32BitIndexMath(t) ? at::kInt : at::kLong;
}

static Tensor _segment_reduce_forward_mps_impl(ReductionType reduction,
                                               const Tensor& data,
                                               const Tensor& lengths_cumsum,
                                               const Tensor& lengths_2d,
                                               int64_t segment_count,
                                               int64_t axis,
                                               const std::optional<Scalar>& initial) {
  auto outer_shape = data.sizes().vec();
  outer_shape[axis] = segment_count;
  auto output = at::empty(outer_shape, data.options());
  if (output.numel() == 0) {
    return output;
  }

  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  int64_t inner_offset = c10::multiply_integers(data.sizes().begin() + axis + 1,
                                                data.sizes().end());
  int64_t data_size_axis = data.size(axis);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_forward_" + scalarToMetalTypeString(data) + "_" +
          scalarToMetalTypeString(lengths_cumsum.scalar_type()) + "_" +
          scalarToMetalTypeString(lengths_2d.scalar_type()) + "_" + reductionToMetalString(reduction);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_mps", [&] {
        SegmentReduceParams params;
        params.segment_count = segment_count;
        params.outer_offset = outer_offset;
        params.inner_offset = inner_offset;
        params.data_size_axis = data_size_axis;
        params.is_initial_set = initial.has_value();
        scalar_t initial_val = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(0);
        mtl_setArgs(computeEncoder,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    params,
                    initial_val);
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

static Tensor _segment_reduce_lengths_mps_kernel(ReductionType reduction,
                                                 const Tensor& data,
                                                 const Tensor& lengths,
                                                 int64_t axis,
                                                 const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_lengths_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());

  int64_t segment_count = lengths.size(-1);
  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  const auto idx_dtype = mps_index_dtype(data);
  auto lengths_2d = lengths.reshape({outer_offset, segment_count}).contiguous();
  auto zeros = at::zeros({outer_offset, 1}, lengths_2d.options().dtype(idx_dtype));
  auto lengths_cumsum = at::cat({zeros, lengths_2d.to(idx_dtype)}, 1).cumsum(1).contiguous();
  return _segment_reduce_forward_mps_impl(reduction, data, lengths_cumsum, lengths_2d, segment_count, axis, initial);
}

static Tensor _segment_reduce_offsets_mps_kernel(ReductionType reduction,
                                                 const Tensor& data,
                                                 const Tensor& offsets,
                                                 int64_t axis,
                                                 const std::optional<Scalar>& initial) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "_segment_reduce_offsets_mps_kernel: only float, half, bfloat16 are supported, got ",
              data.scalar_type());
  int64_t segment_count = offsets.size(-1) - 1;
  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  const auto idx_dtype = mps_index_dtype(data);
  auto lengths_cumsum = offsets.reshape({outer_offset, segment_count + 1}).to(idx_dtype).contiguous();
  auto lengths_2d = lengths_cumsum;
  return _segment_reduce_forward_mps_impl(reduction, data, lengths_cumsum, lengths_2d, segment_count, axis, initial);
}

static Tensor _segment_reduce_backward_mps_impl(const Tensor& grad,
                                                const Tensor& output,
                                                const Tensor& data,
                                                ReductionType reduction,
                                                const Tensor& lengths_cumsum,
                                                const Tensor& lengths_2d,
                                                int64_t segment_count,
                                                int64_t axis,
                                                const std::optional<Scalar>& initial) {
  auto grad_input = at::zeros(data.sizes(), grad.options());
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  int64_t inner_offset = c10::multiply_integers(data.sizes().begin() + axis + 1,
                                                data.sizes().end());
  int64_t data_size_axis = data.size(axis);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "segment_reduce_backward_" + scalarToMetalTypeString(data) + "_" +
          scalarToMetalTypeString(lengths_cumsum.scalar_type()) + "_" +
          scalarToMetalTypeString(lengths_2d.scalar_type()) + "_" + reductionToMetalString(reduction);
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(kernel);
      getMPSProfiler().beginProfileKernel(pso, kernel, {data});
      [computeEncoder setComputePipelineState:pso];
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, data.scalar_type(), "segment_reduce_backward_mps", [&] {
        SegmentReduceParams params;
        params.segment_count = segment_count;
        params.outer_offset = outer_offset;
        params.inner_offset = inner_offset;
        params.data_size_axis = data_size_axis;
        params.is_initial_set = false;
        scalar_t initial_prod_value = initial.has_value() ? initial.value().to<scalar_t>() : scalar_t(1);
        mtl_setArgs(computeEncoder,
                    grad_input,
                    grad,
                    output,
                    data.contiguous(),
                    lengths_cumsum,
                    lengths_2d,
                    params,
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
  int64_t segment_count = lengths.size(-1);
  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  const auto idx_dtype = mps_index_dtype(data);
  auto lengths_2d = lengths.reshape({outer_offset, segment_count}).contiguous();
  auto cumsum_zeros = at::zeros({outer_offset, 1}, lengths_2d.options().dtype(idx_dtype));
  auto lengths_cumsum = at::cat({cumsum_zeros, lengths_2d.to(idx_dtype)}, 1).cumsum(1).contiguous();
  return _segment_reduce_backward_mps_impl(
      grad, output, data, reduction, lengths_cumsum, lengths_2d, segment_count, axis, initial);
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

  int64_t segment_count = offsets.size(-1) - 1;
  int64_t outer_offset = c10::multiply_integers(data.sizes().begin(),
                                                data.sizes().begin() + axis);
  const auto idx_dtype = mps_index_dtype(data);
  auto lengths_cumsum = offsets.reshape({outer_offset, segment_count + 1}).to(idx_dtype).contiguous();
  auto lengths_2d = lengths_cumsum;
  return _segment_reduce_backward_mps_impl(
      grad, output, data, reduction, lengths_cumsum, lengths_2d, segment_count, axis, initial);
}

} // namespace mps

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &mps::_segment_reduce_lengths_mps_kernel);
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &mps::_segment_reduce_lengths_backward_mps_kernel);

REGISTER_DISPATCH(_segment_reduce_offsets_stub, &mps::_segment_reduce_offsets_mps_kernel);
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &mps::_segment_reduce_offsets_backward_mps_kernel);

} // namespace at::native
