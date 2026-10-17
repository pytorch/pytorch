#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/TensorUtils.h>
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
Tensor segment_reduce_mps(ReductionType reduction,
                          const Tensor& data,
                          const Tensor& metadata,
                          int64_t axis,
                          const std::optional<Scalar>& initial) {
  TORCH_CHECK(supportedFloatingType(data), "segment_reduce(): MPS supports float32, float16 and bfloat16 data");
  checkScalarTypes("segment_reduce", TensorArg(metadata, IsOffsets ? "offsets" : "lengths", 3), {kInt, kLong});
  Tensor offsets = metadata;
  if constexpr (!IsOffsets) {
    auto shape = metadata.sizes().vec();
    shape[axis] = 1;
    // Widen before scanning so int32 lengths cannot overflow their prefix sums.
    offsets = at::cat({at::zeros(shape, metadata.options()), metadata}, axis).cumsum(axis, kLong);
  }
  SegmentReduceParams p{1,
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
  for (const auto d : c10::irange(axis + 1, data.dim())) {
    p.inner *= data.size(d);
  }
  auto shape = data.sizes().vec();
  shape[axis] = p.segments;
  auto output = at::empty(shape, data.options());
  if (output.numel() == 0) {
    return output;
  }
  // Amortize threadgroup reduction overhead once contiguous segments span a SIMD group.
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
        mtl_setArgs(encoder, data, output, offsets, p, base);
        if (parallel) {
          // Eight SIMD groups per segment, with one partial per SIMD group.
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

} // namespace

REGISTER_MPS_DISPATCH(_segment_reduce_lengths_stub, &segment_reduce_mps<false>)
REGISTER_MPS_DISPATCH(_segment_reduce_offsets_stub, &segment_reduce_mps<true>)

} // namespace at::native
