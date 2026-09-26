#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/SegmentReduce.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/SegmentReduce.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include <algorithm>
#include <limits>

namespace at::native {
namespace {
using namespace mps;
using c10::metal::SegmentReduceParams;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SegmentReduce_metallib.h>
#endif

const char* reduction_name(ReductionType reduction) {
  switch (reduction) {
    case ReductionType::MAX:
      return "Max";
    case ReductionType::MEAN:
      return "Mean";
    case ReductionType::MIN:
      return "Min";
    case ReductionType::SUM:
      return "Sum";
    case ReductionType::PROD:
      return "Prod";
  }
  TORCH_CHECK(false, "segment_reduce(): unsupported reduction");
}

template <bool IsOffsets>
Tensor segment_offsets(const Tensor& data, const Tensor& metadata, int64_t axis) {
  TORCH_CHECK(data.scalar_type() == kFloat || data.scalar_type() == kHalf || data.scalar_type() == kBFloat16,
              "segment_reduce(): MPS supports float32, float16 and bfloat16 data");
  TORCH_CHECK(data.device() == metadata.device(),
              "segment_reduce(): data and lengths/offsets must be on the same device");
  TORCH_CHECK(metadata.scalar_type() == kInt || metadata.scalar_type() == kLong,
              "segment_reduce(): lengths/offsets must have int32 or int64 dtype");
  TORCH_CHECK(metadata.dim() == axis + 1 && data.dim() > axis,
              "segment_reduce(): Expected axis to be the last dimension of lengths/offsets");
  for (const auto d : c10::irange(axis)) {
    TORCH_CHECK(metadata.size(d) == data.size(d),
                "segment_reduce(): lengths/offsets must match data dimensions before axis");
  }
  if constexpr (IsOffsets) {
    TORCH_CHECK(metadata.size(axis) > 0, "segment_reduce(): offsets must contain at least one element along axis");
    return metadata;
  } else {
    auto shape = metadata.sizes().vec();
    shape[axis] = 1;
    // Widen before scanning so int32 lengths cannot overflow their prefix sums.
    return at::cat({at::zeros(shape, metadata.options()), metadata}, axis).cumsum(axis, kLong);
  }
}

SegmentReduceParams segment_params(const Tensor& data,
                                   const Tensor& offsets,
                                   int64_t axis,
                                   const std::optional<Scalar>& initial) {
  SegmentReduceParams p{1,
                        1,
                        static_cast<uint64_t>(offsets.size(axis) - 1),
                        static_cast<uint64_t>(data.size(axis)),
                        0.0f,
                        initial.has_value()};
  if (initial.has_value()) {
    if (data.scalar_type() == kHalf) {
      p.initial = static_cast<float>(initial->to<Half>());
    } else if (data.scalar_type() == kBFloat16) {
      p.initial = static_cast<float>(initial->to<BFloat16>());
    } else {
      p.initial = initial->toFloat();
    }
  }
  for (const auto d : c10::irange(axis)) {
    p.outer *= data.size(d);
  }
  for (const auto d : c10::irange(axis + 1, data.dim())) {
    p.inner *= data.size(d);
  }
  return p;
}

Tensor validate_offsets(const Tensor& offsets, const SegmentReduceParams& p) {
  auto valid = at::empty({static_cast<int64_t>(p.outer)}, offsets.options().dtype(kInt));
  if (p.outer == 0) {
    return valid;
  }
  auto* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto pipeline = lib.getPipelineStateForFunc("segment_validate_" + scalarToMetalTypeString(offsets));
      [encoder setComputePipelineState:pipeline];
      for (uint64_t base = 0; base < p.outer;) {
        const auto count = std::min<uint64_t>(p.outer - base, std::numeric_limits<uint32_t>::max());
        mtl_setArgs(encoder, offsets, valid, p, stream->getErrorBuffer(), base);
        mtl_dispatch1DJob(encoder, pipeline, count);
        base += count;
      }
    }
  });
  return valid;
}

template <bool IsOffsets>
Tensor segment_reduce_mps(ReductionType reduction,
                          const Tensor& data,
                          const Tensor& metadata,
                          int64_t axis,
                          const std::optional<Scalar>& initial) {
  const auto offsets = segment_offsets<IsOffsets>(data, metadata, axis);
  const auto p = segment_params(data, offsets, axis, initial);
  auto shape = data.sizes().vec();
  shape[axis] = p.segments;
  auto output = at::empty(shape, data.options());
  const auto valid = validate_offsets(offsets, p);
  if (output.numel() == 0) {
    return output;
  }
  const bool parallel = p.inner == 1 && p.axis_size / p.segments >= 32;
  const auto name = fmt::format("segment_{}_{}_{}_{}",
                                parallel ? "parallel" : "serial",
                                scalarToMetalTypeString(data),
                                scalarToMetalTypeString(offsets),
                                reduction_name(reduction));
  auto* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto pipeline = lib.getPipelineStateForFunc(name);
      [encoder setComputePipelineState:pipeline];
      for (uint64_t base = 0; base < static_cast<uint64_t>(output.numel());) {
        const auto count = std::min<uint64_t>(output.numel() - base, std::numeric_limits<uint32_t>::max());
        mtl_setArgs(encoder, data, output, offsets, valid, p, base);
        if (parallel) {
          [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
          mtl_dispatch1DJob(encoder, pipeline, count);
        }
        base += count;
      }
    }
  });
  return output;
}

template <bool IsOffsets>
Tensor segment_reduce_backward_mps(const Tensor& grad,
                                   const Tensor& output,
                                   const Tensor& data,
                                   ReductionType reduction,
                                   const Tensor& metadata,
                                   int64_t axis,
                                   const std::optional<Scalar>& initial) {
  const auto offsets = segment_offsets<IsOffsets>(data, metadata, axis);
  const auto p = segment_params(data, offsets, axis, initial);
  auto shape = data.sizes().vec();
  shape[axis] = p.segments;
  TORCH_CHECK(output.sizes() == IntArrayRef(shape) && grad.sizes() == output.sizes(),
              "segment_reduce_backward(): grad and output must match the reduced shape");
  TORCH_CHECK(grad.scalar_type() == data.scalar_type() && output.scalar_type() == data.scalar_type(),
              "segment_reduce_backward(): grad, output and data must have the same dtype");
  TORCH_CHECK(grad.device() == data.device() && output.device() == data.device(),
              "segment_reduce_backward(): grad, output and data must be on the same device");
  auto grad_input = at::zeros(data.sizes(), data.options());
  const auto valid = validate_offsets(offsets, p);
  if (output.numel() == 0 || data.numel() == 0) {
    return grad_input;
  }
  const auto prod_prefix =
      reduction == ReductionType::PROD ? at::empty(data.sizes(), data.options().dtype(kFloat)) : grad_input;
  const bool parallel = p.inner == 1 && p.axis_size / p.segments >= 1024;
  const auto name = fmt::format("segment_backward_{}{}_{}_{}",
                                parallel ? "parallel_" : "",
                                scalarToMetalTypeString(data),
                                scalarToMetalTypeString(offsets),
                                reduction_name(reduction));
  auto* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      auto pipeline = lib.getPipelineStateForFunc(name);
      [encoder setComputePipelineState:pipeline];
      for (uint64_t base = 0; base < static_cast<uint64_t>(output.numel());) {
        const auto count = std::min<uint64_t>(output.numel() - base, std::numeric_limits<uint32_t>::max());
        mtl_setArgs(encoder, grad, output, data, grad_input, offsets, valid, p, base, prod_prefix);
        if (parallel) {
          [encoder dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
          mtl_dispatch1DJob(encoder, pipeline, count);
        }
        base += count;
      }
    }
  });
  return grad_input;
}

} // namespace

REGISTER_MPS_DISPATCH(_segment_reduce_lengths_stub, &segment_reduce_mps<false>)
REGISTER_MPS_DISPATCH(_segment_reduce_offsets_stub, &segment_reduce_mps<true>)
REGISTER_MPS_DISPATCH(_segment_reduce_lengths_backward_stub, &segment_reduce_backward_mps<false>)
REGISTER_MPS_DISPATCH(_segment_reduce_offsets_backward_stub, &segment_reduce_backward_mps<true>)

} // namespace at::native
