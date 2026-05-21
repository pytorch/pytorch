//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/bincount_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Bincount_metallib.h>
#endif

// Bool isn't a supported index dtype for bincount on any backend. Historical
// note from the legacy MPSGraph implementation: passing a bool input crashed
// in MPSGraphUtilities.mm with "'mps.scatter' op operand #2 must be tensor of
// int values, but got 'tensor<5xi1>'". The native Metal path here doesn't
// register a bool kernel either, so the check stays.
static Tensor bincount_mps_unweighted(const Tensor& self, int64_t nbins) {
  // Two-stage: accumulate into a uint32 atomic scratch buffer, then widen to
  // int64 with a small fused kernel on the SAME encoder. uint32 + fused
  // widening is measurably faster on heavily-skewed inputs than the
  // alternative AtomicType<long>, which would double the per-thread atomic
  // ops; uint32 is safe because the wrapper asserts numel <= UINT32_MAX and
  // count(bin) <= numel.
  Tensor counts_u32 = at::zeros({nbins}, self.options().dtype(kInt));
  Tensor output = at::empty({nbins}, self.options().dtype(kLong));

  const std::string add_key = "bincount_unweighted_" + scalarToMetalTypeString(self);
  const std::string widen_key = "bincount_widen_uint_to_long";
  const int64_t self_stride = self.stride(0);
  const uint64_t numel = static_cast<uint64_t>(self.numel());
  const uint64_t nbins_u = static_cast<uint64_t>(nbins);

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> add_pso = lib.getPipelineStateForFunc(add_key);
      id<MTLComputePipelineState> widen_pso = lib.getPipelineStateForFunc(widen_key);

      getMPSProfiler().beginProfileKernel(add_pso, add_key, false);
      [encoder setComputePipelineState:add_pso];
      mtl_setArgs(encoder, self, counts_u32, self_stride);
      mtl_dispatch1DJob(encoder, add_pso, static_cast<NSUInteger>(numel));
      getMPSProfiler().endProfileKernel(add_pso);

      // Metal serialises back-to-back compute dispatches on the same encoder,
      // so the widen below sees the completed accumulator without an
      // explicit barrier.
      getMPSProfiler().beginProfileKernel(widen_pso, widen_key, false);
      [encoder setComputePipelineState:widen_pso];
      mtl_setArgs(encoder, counts_u32, output);
      mtl_dispatch1DJob(encoder, widen_pso, static_cast<NSUInteger>(nbins_u));
      getMPSProfiler().endProfileKernel(widen_pso);
    }
  });

  return output;
}

// Maps PyTorch weight dtypes to the Metal kernel suffix used by the
// bincount_weighted<IDX_T, T> template. Float/Int/Long all have a
// native (non-CAS) AtomicType implementation and run fast on contended
// workloads. Half/BFloat16 are accumulated in Float (and cast back) -- the
// `AtomicType<half>`/`<bfloat>` CAS-based primitives measure ~75x slower
// than `atomic<float>` under contention, which dominates real bincount
// inputs. Narrow integer types (Char/Short/Byte) and Double upcast to Int
// and return Int, preserving the dtype contract the prior MPSGraph path
// established (see test_bincount_reduction).
struct WeightedDispatch {
  std::string kernel_suffix; // suffix in the metal kernel name
  ScalarType kernel_dtype; // dtype the kernel accumulates in
  ScalarType output_dtype; // dtype the wrapper returns (cast back if != kernel_dtype)
};

static WeightedDispatch dispatch_for_weight_dtype(ScalarType weight_dtype) {
  switch (weight_dtype) {
    case kFloat:
      return {"float", kFloat, kFloat};
    case kHalf:
      return {"float", kFloat, kHalf}; // cast in, accumulate float, cast out
    case kBFloat16:
      return {"float", kFloat, kBFloat16}; // ditto
    case kInt:
      return {"int", kInt, kInt};
    case kLong:
      return {"long", kLong, kLong}; // native AtomicType<long> support
    case kChar:
    case kShort:
    case kByte:
    case kDouble:
      return {"int", kInt, kInt};
    default:
      TORCH_CHECK(false, "bincount: unsupported weights dtype on MPS: ", weight_dtype);
  }
}

static Tensor bincount_mps_weighted(const Tensor& self, const Tensor& weights, int64_t nbins) {
  const auto dispatch = dispatch_for_weight_dtype(weights.scalar_type());
  Tensor weights_for_kernel =
      (weights.scalar_type() == dispatch.kernel_dtype) ? weights : weights.to(dispatch.kernel_dtype);
  Tensor accum = at::zeros({nbins}, weights.options().dtype(dispatch.kernel_dtype));

  const std::string key = "bincount_weighted_" + dispatch.kernel_suffix + "_" + scalarToMetalTypeString(self);
  const int64_t self_stride = self.stride(0);
  const int64_t weights_stride = weights_for_kernel.stride(0);
  const uint64_t numel = static_cast<uint64_t>(self.numel());

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);
      getMPSProfiler().beginProfileKernel(pso, key, false);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, weights_for_kernel, accum, self_stride, weights_stride);
      mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(numel));
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  if (dispatch.output_dtype != dispatch.kernel_dtype) {
    return accum.to(dispatch.output_dtype);
  }
  return accum;
}

Tensor _bincount_mps(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  TORCH_CHECK(c10::isIntegralType(self.scalar_type(), /*includesBool=*/true));
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");

  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong, std::nullopt, kMPS, std::nullopt);
  }
  TORCH_CHECK(self.dim() == 1, "bincount only supports 1-d non-negative integral inputs.");
  TORCH_CHECK(self.scalar_type() != kBool, "bincount is not supported for Bool");

  bool has_weights = weights.defined();
  TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))),
              "weights should be 1-d and have the same length as input");
  // Per-bin counts are accumulated in uint32 atomics; capping numel at
  // UINT32_MAX prevents any individual count from overflowing (since
  // count(bin) <= numel by construction). This is ~4.29 billion elements,
  // which is well beyond any realistic 1-d bincount input.
  TORCH_CHECK(self.numel() <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
              "bincount on MPS supports inputs with at most 2^32-1 elements");

  // Single fused aminmax reduces both bounds in one MPS dispatch round-trip
  // rather than separate self.max() + self.min() calls.
  const auto [input_min_t, input_max_t] = at::aminmax(self);
  const int64_t input_min = input_min_t.item<int64_t>();
  const int64_t input_max = input_max_t.item<int64_t>();
  TORCH_CHECK(input_min >= 0, "bincount only supports 1-d non-negative integral inputs.");

  const int64_t nbins = std::max(input_max + 1, minlength);

  if (has_weights) {
    return bincount_mps_weighted(self, weights, nbins);
  } else {
    return bincount_mps_unweighted(self, nbins);
  }
}

} // namespace at::native
