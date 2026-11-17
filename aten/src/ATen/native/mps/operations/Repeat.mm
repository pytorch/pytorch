//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Repeat.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/repeat_interleave_native.h>
#include <ATen/ops/repeat_native.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Repeat_metallib.h>
#endif

Tensor permute_mps(const Tensor& self, IntArrayRef dims) {
  auto nDims = self.dim();
  TORCH_CHECK(dims.size() == (size_t)nDims, "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

// The 2-D gather kernel beats the strided copy on small inputs and transposed reads.
static bool use_repeat_kernel(const Tensor& self, IntArrayRef repeats, const Tensor& result) {
  using namespace mps;
  // Only 2-D inputs; when every repeat is 1 a plain copy is faster.
  if (self.dim() != 2 || repeats.size() != 2 || result.numel() == self.numel()) {
    return false;
  }
  // copy_ handles lazy conj/neg flags and tensors too large for int32 offsets.
  if (self.is_conj() || self.is_neg() || !offsetsFitIn<int32_t>(self, result)) {
    return false;
  }
  // Every dtype but int64 takes the kernel.
  if (self.scalar_type() != kLong) {
    return true;
  }
  // int64 with a non-unit row stride is faster through copy_ unless transposed.
  const bool transposed = self.stride(0) < self.stride(1);
  return self.stride(1) <= 1 || transposed;
}

Tensor repeat_mps(const Tensor& self, IntArrayRef repeats) {
  using namespace mps;
  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
              "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor result = at::empty(target_size, self.options());
  if (result.numel() == 0) {
    return result;
  }

  if (use_repeat_kernel(self, repeats, result)) {
    // A size-1 dim never adds an offset, but its stride can exceed int32.
    const std::array<int32_t, 4> sizes_strides = {
        c10::checked_convert<int32_t>(self.size(1), "int32_t"),
        c10::checked_convert<int32_t>(self.size(0), "int32_t"),
        c10::checked_convert<int32_t>(self.size(1) == 1 ? 0 : self.stride(1), "int32_t"),
        c10::checked_convert<int32_t>(self.size(0) == 1 ? 0 : self.stride(0), "int32_t")};
    auto* stream = getCurrentMPSStream();
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        const auto name = fmt::format("repeat_2d_{}", self.element_size());
        auto encoder = stream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc(name);
        getMPSProfiler().beginProfileKernel(pso, name, {self}, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, self, result, sizes_strides);
        const auto max_threads = [pso maxTotalThreadsPerThreadgroup];
        // One SIMD group per row tiles transposed reads across neighboring rows.
        const auto width = self.stride(0) < self.stride(1) ? [pso threadExecutionWidth] : max_threads;
        const auto tg_x = std::min<NSUInteger>(result.size(1), width);
        const auto tg_y = std::min<NSUInteger>(result.size(0), max_threads / tg_x);
        [encoder dispatchThreads:MTLSizeMake(result.size(1), result.size(0), 1)
            threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
        getMPSProfiler().endProfileKernel(pso, stream);
      }
    });
    return result;
  }

  auto unfolded = result;
  for (const auto dim : c10::irange(repeats.size())) {
    unfolded = unfolded.unfold(dim, padded_size[dim], padded_size[dim]);
  }
  unfolded.copy_(self);
  return result;
}

Tensor repeat_interleave_mps(const Tensor& repeat, std::optional<int64_t> output_size) {
  TORCH_CHECK(repeat.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  std::string scalar_type;
  if (repeat.scalar_type() == kInt) {
    scalar_type = "int32_t";
  } else if (repeat.scalar_type() == kLong) {
    scalar_type = "int64_t";
  } else {
    TORCH_CHECK(false, "repeats has to be Long or Int tensor");
  }
  if (repeat.size(0) == 0) {
    return at::empty_like(repeat, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  Tensor cumsum = repeat.cumsum(0);
  int64_t total = 0;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
    TORCH_CHECK((repeat >= 0).all().item<uint8_t>(), "repeats can not be negative");
  }

  auto result = at::empty({total}, repeat.options());

  MPSStream* mpsStream = getCurrentMPSStream();
  at::mps::dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pipelineState = lib.getPipelineStateForFunc(fmt::format("repeat_interleave_{}", scalar_type));

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(pipelineState, "repeat_interleave:" + scalar_type, false, mpsStream);

      [computeEncoder setComputePipelineState:pipelineState];
      mps::mtl_setArgs(computeEncoder, repeat, cumsum, result, repeat.stride(0));
      mps::mtl_dispatch1DJob(computeEncoder, pipelineState, repeat.size(0));

      getMPSProfiler().endProfileKernel(pipelineState, mpsStream);
    }
  });
  return result;
}

} // namespace at::native
