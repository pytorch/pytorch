#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SegmentReduce.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/cumsum.h>
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

// Both entry points are normalized to offsets so the kernels only have to
// implement one addressing scheme.
static Tensor lengths_to_offsets(const Tensor& lengths) {
  auto sizes = lengths.sizes().vec();
  sizes.back() = 1;
  const auto zero = at::zeros(sizes, lengths.options().dtype(kLong));
  return at::cat({zero, lengths.to(kLong).cumsum(-1)}, -1).contiguous();
}

static SegmentReduceParams make_params(ReductionType reduction,
                                       const Tensor& data,
                                       const Tensor& output,
                                       int64_t axis,
                                       int64_t segment_count,
                                       const std::optional<Scalar>& initial) {
  int64_t outer_offset = 1, inner_offset = 1;
  for (const auto d : c10::irange(axis)) {
    outer_offset *= output.size(d);
  }
  for (const auto d : c10::irange(axis + 1, output.dim())) {
    inner_offset *= output.size(d);
  }

  return SegmentReduceParams{
      .outer_offset = outer_offset,
      .inner_offset = inner_offset,
      .segment_count = segment_count,
      .data_stride_axis = data.stride(axis),
      .data_size_axis = data.size(axis),
      .output_stride_axis = output.stride(axis),
      .output_size_axis = output.size(axis),
      .offsets_stride_axis = 1,
      .offsets_size_axis = segment_count + 1,
      .initial = initial.has_value() ? initial.value().to<float>() : 0.0f,
      .has_initial = initial.has_value(),
      .reduction = static_cast<int>(reduction),
  };
}

static void launch(const std::string& kernel,
                   const std::initializer_list<Tensor>& buffers,
                   const SegmentReduceParams& params) {
  const auto numThreads = static_cast<uint32_t>(params.outer_offset * params.segment_count * params.inner_offset);
  if (numThreads == 0) {
    return;
  }
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto pso = lib.getPipelineStateForFunc(kernel);
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        [computeEncoder setComputePipelineState:pso];
        unsigned idx = 0;
        for (const auto& t : buffers) {
          mtl_setBuffer(computeEncoder, t, idx++);
        }
        mtl_setBytes(computeEncoder, params, idx);
        mtl_dispatch1DJob(computeEncoder, pso, numThreads);
      }
    });
  }
}

static Tensor segment_reduce_offsets_mps(ReductionType reduction,
                                         const Tensor& data,
                                         const Tensor& offsets,
                                         int64_t axis,
                                         const std::optional<Scalar>& initial) {
  axis = offsets.dim() - 1;
  const auto segment_count = offsets.size(axis) - 1;
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  const auto offsets_l = offsets.scalar_type() == kLong ? offsets.contiguous() : offsets.to(kLong).contiguous();
  const auto params = make_params(reduction, data, output, axis, segment_count, initial);
  launch("segment_reduce_" + scalarToMetalTypeString(data), {data, offsets_l, output}, params);
  return output;
}

static Tensor segment_reduce_lengths_mps(ReductionType reduction,
                                         const Tensor& data,
                                         const Tensor& lengths,
                                         int64_t axis,
                                         const std::optional<Scalar>& initial) {
  return segment_reduce_offsets_mps(reduction, data, lengths_to_offsets(lengths), axis, initial);
}

static Tensor segment_reduce_offsets_backward_mps(const Tensor& grad,
                                                  const Tensor& output,
                                                  const Tensor& data,
                                                  ReductionType reduction,
                                                  const Tensor& offsets,
                                                  int64_t axis,
                                                  const std::optional<Scalar>& initial) {
  axis = offsets.dim() - 1;
  const auto segment_count = offsets.size(axis) - 1;
  auto grad_input = at::zeros(data.sizes(), grad.options());

  const auto offsets_l = offsets.scalar_type() == kLong ? offsets.contiguous() : offsets.to(kLong).contiguous();
  const auto params = make_params(reduction, data, output, axis, segment_count, initial);
  launch(
      "segment_reduce_backward_" + scalarToMetalTypeString(data), {grad_input, grad, output, data, offsets_l}, params);
  return grad_input;
}

static Tensor segment_reduce_lengths_backward_mps(const Tensor& grad,
                                                  const Tensor& output,
                                                  const Tensor& data,
                                                  ReductionType reduction,
                                                  const Tensor& lengths,
                                                  int64_t axis,
                                                  const std::optional<Scalar>& initial) {
  return segment_reduce_offsets_backward_mps(grad, output, data, reduction, lengths_to_offsets(lengths), axis, initial);
}

} // namespace mps

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &mps::segment_reduce_lengths_mps)
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &mps::segment_reduce_offsets_mps)
REGISTER_DISPATCH(_segment_reduce_lengths_backward_stub, &mps::segment_reduce_lengths_backward_mps)
REGISTER_DISPATCH(_segment_reduce_offsets_backward_stub, &mps::segment_reduce_offsets_backward_mps)

} // namespace at::native
