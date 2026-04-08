//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Fill.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ConstantKernel_metallib.h>
#endif

static void fill_mps_kernel(TensorIterator& iter, const Scalar& value) {
  using namespace mps;
  if (iter.numel() == 0) {
    return;
  }

  // Metal compute kernels use uint (32-bit) thread indices; decompose large
  // tensors into chunks that fit in 32-bit indexing.
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      fill_mps_kernel(sub_iter, value);
    }
    return;
  }

  const Tensor& self = iter.tensor(0);
  const auto dtype = self.scalar_type();
  const auto stream = getCurrentMPSStream();
  const auto type_str = scalarToMetalTypeString(dtype);
  const bool can_fill_linearly = self.is_non_overlapping_and_dense();

  // For tensors with gaps or overlaps (e.g. stride-2 slices) use a 2D strided
  // kernel: tid.y indexes dim 0 directly (no division), tid.x is the linear
  // index for the remaining dims.  Consecutive threads in x write consecutive
  // addresses in the innermost dimension, giving coalesced writes.
  if (!can_fill_linearly) {
    auto fillPSO = lib.getPipelineStateForFunc(fmt::format("fill_scalar_strided_{}", type_str));
    const int64_t dim0_size = iter.ndim() > 0 ? iter.shape()[0] : 1;
    const int64_t inner_numel = iter.numel() / dim0_size;
    const uint32_t ndim = static_cast<uint32_t>(iter.ndim());
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        auto mpsScalar = getMPSScalar(value, dtype);
        [computeEncoder setComputePipelineState:fillPSO];
        bind_iter_tensors(computeEncoder, iter);
        mtl_setArgs<1>(computeEncoder, mpsScalar, iter.shape(), iter.strides(0), ndim);
        const NSUInteger maxTG = fillPSO.maxTotalThreadsPerThreadgroup;
        const MTLSize tgSize = MTLSizeMake(std::min(maxTG, (NSUInteger)inner_numel), 1, 1);
        const MTLSize gridSize = MTLSizeMake(inner_numel, dim0_size, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
      }
    });
    return;
  }

  // Single-byte dtypes (bool, uint8, int8) use vec4 kernels that fill
  // 4 elements per thread; all others fill 1 element per thread.
  const bool is_byte_type = self.element_size() == 1;
  const uint32_t numel = static_cast<uint32_t>(iter.numel());
  const int64_t threads = is_byte_type ? (numel + 3) / 4 : numel;

  auto fillPSO = lib.getPipelineStateForFunc(fmt::format("fill_scalar_dense_{}", type_str));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto mpsScalar = getMPSScalar(value, dtype);
      [computeEncoder setComputePipelineState:fillPSO];
      bind_iter_tensors(computeEncoder, iter);
      mtl_setArgs<1>(computeEncoder, mpsScalar, numel);
      mtl_dispatch1DJob(computeEncoder, fillPSO, threads);
    }
  });
}

REGISTER_DISPATCH(fill_stub, &fill_mps_kernel);

} // namespace at::native
