#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReduceOps_metallib.h>
#endif

template <bool is_offsets>
static Tensor segment_reduce_mps(ReductionType reduction,
                                 const Tensor& data,
                                 const Tensor& lengths_or_offsets,
                                 int64_t axis,
                                 const std::optional<Scalar>& initial) {
  using namespace mps;
  using namespace std::string_view_literals;
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(data),
                              "segment_reduce(): MPS supports float32, float16 and bfloat16 data, but got ",
                              data.scalar_type());
  auto index_arg = TensorArg(lengths_or_offsets, is_offsets ? "offsets" : "lengths", 3);
  checkScalarTypes("segment_reduce", index_arg, {kLong, kInt});
  const auto offsets =
      is_offsets ? lengths_or_offsets.to(kLong) : at::constant_pad_nd(lengths_or_offsets, {1, 0}).cumsum(axis, kLong);
  const auto outer = c10::multiply_integers(data.sizes().slice(0, axis));
  const auto inner = c10::multiply_integers(data.sizes().slice(axis + 1));
  const auto segments = offsets.size(axis) - 1;
  auto shape = data.sizes().vec();
  shape[axis] = segments;
  auto output = at::empty(shape, data.options());
  if (output.numel() == 0) {
    return output;
  }
  const bool parallel = inner == 1 && data.size(axis) / segments >= SEGMENT_PARALLEL_MIN_LEN;
  const bool use_u32 = offsetsFitIn<int32_t>(data, offsets, output);
  constexpr std::string_view reduction_names[] = {"max", "mean", "min", "sum", "prod"}; // in ReductionType order
  const auto kernel = fmt::format("segment_{}_{}{}_{}",
                                  reduction_names[static_cast<size_t>(reduction)],
                                  scalarToMetalTypeString(data),
                                  mtlIdxSuffix(use_u32),
                                  parallel ? "parallel"sv : "serial"sv);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc(kernel);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtlDispatchByIndexWidth<uint32_t, uint64_t>(use_u32, [&](auto idx_tag) {
        using idx_t = typename decltype(idx_tag)::type;
        const SegmentReduceParams<idx_t> params{.inner = static_cast<idx_t>(inner),
                                                .segments = static_cast<idx_t>(segments),
                                                .axis_size = static_cast<idx_t>(data.size(axis)),
                                                .initial = initial.value_or(0).toFloat(),
                                                .has_initial = initial.has_value()};
        mtl_setArgs(encoder, data, output, offsets, params);
      });
      if (parallel) {
        // The kernel strides by INNER_TG_SIZE, so each segment must get exactly one full threadgroup
        [encoder dispatchThreadgroups:MTLSizeMake(1, segments, outer)
                threadsPerThreadgroup:MTLSizeMake(INNER_TG_SIZE, 1, 1)];
      } else {
        mtl_dispatch3DJob(encoder, pso, inner, segments, outer);
      }
    }
  });
  return output;
}

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &segment_reduce_mps<false>)
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &segment_reduce_mps<true>)

} // namespace at::native
