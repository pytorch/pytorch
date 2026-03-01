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

  const Tensor& self = iter.tensor(0);
  const auto dtype = self.scalar_type();
  const auto stream = getCurrentMPSStream();

  // For fill_, every logical element gets the same value, so element order
  // does not matter. Any tensor that is non-overlapping and dense (all storage
  // positions covered exactly once) can be filled by writing linearly to its
  // underlying storage, regardless of its dimensional layout.
  // This covers contiguous tensors AND "effectively-contiguous" views such as
  // transposed or permuted tensors.
  const bool can_fill_linearly = self.is_non_overlapping_and_dense();

  // Use Metal fillBuffer for zero fill and byte-representable fills on
  // linearly-fillable tensors: these map to a single-byte pattern.
  if (can_fill_linearly && !isComplexType(dtype)) {
    if (value.toDouble() == 0.0) {
      stream->fill(getMTLBufferStorage(self), 0, self.nbytes(), self.storage_offset() * self.itemsize());
      return;
    }
    if (dtype == kBool) {
      stream->fill(getMTLBufferStorage(self), (uint8_t)value.toBool(), self.nbytes(), self.storage_offset() * self.itemsize());
      return;
    }
    if (dtype == kByte) {
      stream->fill(getMTLBufferStorage(self), (uint8_t)value.toByte(), self.nbytes(), self.storage_offset() * self.itemsize());
      return;
    }
    if (dtype == kChar) {
      stream->fill(getMTLBufferStorage(self), (uint8_t)(int8_t)value.toChar(), self.nbytes(), self.storage_offset() * self.itemsize());
      return;
    }
  }

  const auto type_str = scalarToMetalTypeString(dtype);

  // For tensors with gaps or overlaps (e.g. stride-2 slices) use a 2D strided
  // kernel: tid.y indexes dim 0 directly (no division), tid.x is the linear
  // index for the remaining dims.  Consecutive threads in x write consecutive
  // addresses in the innermost dimension, giving coalesced writes.
  if (!can_fill_linearly) {
    auto fillPSO = lib.getPipelineStateForFunc(fmt::format("fill_scalar_strided_{}", type_str));
    const int64_t dim0_size = self.dim() > 0 ? self.size(0) : 1;
    const int64_t inner_numel = self.numel() / dim0_size;
    const uint32_t ndim = static_cast<uint32_t>(self.dim());
    int64_t sizes_buf[16], strides_buf[16];
    for (uint32_t i = 0; i < ndim; i++) {
      sizes_buf[i] = self.size(i);
      strides_buf[i] = self.stride(i);
    }
    // Blocks can't capture C arrays; use pointer aliases (safe: dispatch_sync blocks until done).
    const int64_t* sizes_ptr = sizes_buf;
    const int64_t* strides_ptr = strides_buf;
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = stream->commandEncoder();
        auto mpsScalar = getMPSScalar(value, dtype);
        [computeEncoder setComputePipelineState:fillPSO];
        mtl_setArgs(computeEncoder, self, mpsScalar);
        [computeEncoder setBytes:sizes_ptr length:ndim * sizeof(int64_t) atIndex:2];
        [computeEncoder setBytes:strides_ptr length:ndim * sizeof(int64_t) atIndex:3];
        [computeEncoder setBytes:&ndim length:sizeof(uint32_t) atIndex:4];
        const NSUInteger maxTG = fillPSO.maxTotalThreadsPerThreadgroup;
        const MTLSize tgSize = MTLSizeMake(std::min(maxTG, (NSUInteger)inner_numel), 1, 1);
        const MTLSize gridSize = MTLSizeMake(inner_numel, dim0_size, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
      }
    });
    return;
  }

  auto fillPSO = lib.getPipelineStateForFunc(fmt::format("fill_scalar_dense_{}", type_str));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = stream->commandEncoder();
      auto mpsScalar = getMPSScalar(value, dtype);
      [computeEncoder setComputePipelineState:fillPSO];
      mtl_setArgs(computeEncoder, self, mpsScalar);
      mtl_dispatch1DJob(computeEncoder, fillPSO, self.numel());
    }
  });
}

REGISTER_DISPATCH(fill_stub, &fill_mps_kernel);

} // namespace at::native
