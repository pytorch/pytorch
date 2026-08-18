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

// Bool isn't a supported index dtype for bincount on any backend (the dtype
// allowlist in _bincount_mps rejects it). Historical note from the legacy
// MPSGraph implementation: passing a bool input crashed in MPSGraphUtilities.mm
// with "'mps.scatter' op operand #2 must be tensor of int values, but got
// 'tensor<5xi1>'"; the native Metal path here registers no bool kernel either.
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

      // counts_u32 is hazard-tracked tensor storage, so Metal auto-inserts the
      // write->read barrier between these two same-encoder dispatches; no
      // explicit barrier is needed. (Serial-encoder ordering alone does not
      // prevent dispatch overlap.)
      getMPSProfiler().beginProfileKernel(widen_pso, widen_key, false);
      [encoder setComputePipelineState:widen_pso];
      mtl_setArgs(encoder, counts_u32, output);
      mtl_dispatch1DJob(encoder, widen_pso, static_cast<NSUInteger>(nbins_u));
      getMPSProfiler().endProfileKernel(widen_pso);
    }
  });

  return output;
}

// Weighted bincount always accumulates in and returns float32. CPU/CUDA return
// float32 for float32 weights and float64 for every other weight dtype (int,
// half/bfloat16, double); MPS has no float64, so float32 is used throughout.
// This matches the reference type class (float) and its float32 case exactly,
// and is the closest MPS can get to the float64 cases -- with reduced precision
// for large integer-weighted sums, which float64 would preserve. atomic<float>
// is also the only native (non-CAS) Metal float atomic; AtomicType<half>/
// <bfloat> are CAS-based and ~75x slower under contention.
static Tensor bincount_mps_weighted(const Tensor& self, const Tensor& weights, int64_t nbins) {
  Tensor weights_f = weights.scalar_type() == kFloat ? weights : weights.to(kFloat);
  Tensor accum = at::zeros({nbins}, weights.options().dtype(kFloat));

  const std::string key = "bincount_weighted_" + scalarToMetalTypeString(self);
  const int64_t self_stride = self.stride(0);
  const int64_t weights_stride = weights_f.stride(0);
  const uint64_t numel = static_cast<uint64_t>(self.numel());

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(key);
      getMPSProfiler().beginProfileKernel(pso, key, false);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, weights_f, accum, self_stride, weights_stride);
      mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(numel));
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  return accum;
}

Tensor _bincount_mps(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  // isIntegralType(..., includesBool=true) also admits UInt16/UInt32/UInt64,
  // which are creatable on MPS but have no registered kernel; restricting to
  // the five supported index dtypes here gives a clean error instead of a
  // cryptic getPipelineStateForFunc failure (and subsumes the kBool check).
  const auto st = self.scalar_type();
  TORCH_CHECK(st == kByte || st == kChar || st == kShort || st == kInt || st == kLong,
              "bincount only supports int8/int16/int32/int64/uint8 inputs on MPS, got ",
              st);
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");

  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, kLong, std::nullopt, kMPS, std::nullopt);
  }
  TORCH_CHECK(self.dim() == 1, "bincount only supports 1-d non-negative integral inputs.");

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
