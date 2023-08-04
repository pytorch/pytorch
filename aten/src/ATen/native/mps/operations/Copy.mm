//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>

namespace at::native {
namespace mps {

void* pageAlignedBlockPtr(const void* ptr, NSUInteger size, NSUInteger* alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  assert(address >= alignedAddress);
  assert(address + size <= alignedAddress + alignedLength);

  *alignedBlockSize = alignedLength;
  return (void*)alignedAddress;
}

// Copy sourceBuffer into destBuffer, casting sourceBuffer to dst.scalar_type().
// The shapes and dtypes are taken from dst and src, but their storage pointers are not used.
void copy_cast_mps(at::Tensor& dst,
                   const at::Tensor& src,
                   id<MTLBuffer> destBuffer,
                   id<MTLBuffer> sourceBuffer,
                   bool non_blocking = true) {
  using namespace mps;

  using CachedGraph = MPSUnaryCachedGraph;

  MPSStream* stream = getCurrentMPSStream();

  MPSDataType dstDType = getMPSDataType(dst);
  MPSDataType srcDType = getMPSDataType(src);
  MPSShape* dstShape = getMPSShape(dst);
  MPSShape* srcShape = getMPSShape(src);

  @autoreleasepool {
    string key = "copy_cast_mps" + getTensorsStringKey({src, dst});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, src);
      MPSGraphTensor* inputCastTensor = inputTensor;
      if (isFloatingType(src.scalar_type()) && dstDType == MPSDataTypeUInt8) {
        inputCastTensor = [mpsGraph castTensor:inputTensor toType:MPSDataTypeInt32 name:@"cast"];
      }
      MPSGraphTensor* outputTensor = [mpsGraph castTensor:inputCastTensor toType:dstDType name:@"cast"];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    MPSGraphTensorData* srcData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:sourceBuffer
                                                                           shape:srcShape
                                                                        dataType:srcDType] autorelease];
    MPSGraphTensorData* dstData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:destBuffer
                                                                           shape:dstShape
                                                                        dataType:dstDType] autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{cachedGraph->inputTensor_ : srcData};
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{cachedGraph->outputTensor_ : dstData};
    stream->executeMPSGraph(
        cachedGraph->graph(), feeds, results, !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT_ADAPTIVE);
  }
}

static at::Tensor& copy_from_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  auto sameMemFormat =
      src_.is_contiguous(dst_.suggest_memory_format()) && dst_.is_contiguous(dst_.suggest_memory_format());

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* stream = getCurrentMPSStream();
  Tensor dst;
  Tensor src;
  if (!dst_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) {
    dst = at::empty_like(dst_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    dst = dst_;
  }

  auto storage_byte_offset = src_.storage_offset() * src_.itemsize();
  if (!src_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) {
    Tensor emptyShell = Tensor();
    src = gatherViewTensor(src_, emptyShell);
    if (src.has_storage()) {
      storage_byte_offset = 0;
    } else {
      src = src_.expand_as(dst).contiguous();
      storage_byte_offset = src.storage_offset() * src.itemsize();
    }
  } else {
    src = src_;
  }
  id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  size_t dst_tensor_nbytes = dst.nbytes();

  @autoreleasepool {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;

    const void* host_dst = dst.storage().data();
    void* alignedPtr = pageAlignedBlockPtr(host_dst, (NSUInteger)dst_tensor_nbytes, &alignedLength);
    NSUInteger destOffset = (uintptr_t(host_dst) - uintptr_t(alignedPtr));
    // 4 bytes alignment required on macos for blits.
    TORCH_INTERNAL_ASSERT(destOffset % 4 == 0, "Unaligned blit request");

    id<MTLBuffer> destBuffer = [device newBufferWithBytesNoCopy:alignedPtr
                                                         length:alignedLength
                                                        options:options
                                                    deallocator:nil];
    id<MTLBuffer> tmpBuffer = sourceBuffer;
    Tensor tmp;
    bool needsBlit = true;
    if (src_.dtype() != dst.dtype()) {
      if (destOffset == 0 && storage_byte_offset == 0) {
        // Return the casted tensor directly if there's no destination offset
        needsBlit = false;
        tmpBuffer = destBuffer;
      } else if (src.element_size() < dst.element_size()) {
        tmp = at::empty(dst.sizes(), dst.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
        tmpBuffer = getMTLBufferStorage(tmp);
      }
    }

    size_t size_to_copy = src.nbytes();
    // In case of dtype change, first convert src inplace
    if (src_.dtype() != dst.dtype()) {
      copy_cast_mps(dst, src, tmpBuffer, sourceBuffer, non_blocking);
    }

    if (needsBlit) {
      size_to_copy = (size_to_copy / src.element_size()) * dst.element_size();

      // If there's anything wrong with source, we shouldn't return dst_ silently and must error out.
      TORCH_INTERNAL_ASSERT(sourceBuffer && dst_tensor_nbytes > 0);
      uint64_t profile_id =
          getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst, size_to_copy, non_blocking);

      stream->copy_and_sync(
          tmpBuffer, destBuffer, size_to_copy, storage_byte_offset, destOffset, non_blocking, profile_id);
      [destBuffer release];
    }
  }
  if (!dst.is_same(dst_)) {
    dst_.copy_(dst, non_blocking);
  }

  return dst_;
}

// Copies tensor from cpu to mps backed by identical strided-contiguous data
static void copy_to_mps_stride_contig(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  MPSStream* stream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  auto dst_byte_offset = dst.storage_offset() * dst.itemsize();
  auto src_byte_offset = src.storage_offset() * src.itemsize();
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst);
  const size_t size_to_copy = src.nbytes();
  const void* host_src = static_cast<const char*>(src.storage().data()) + src_byte_offset;

  TORCH_INTERNAL_ASSERT(src.dtype() == dst.dtype() && src.strides() == dst.strides() && is_dense_in_storage(src));

  @autoreleasepool {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;
    NSUInteger sourceOffset = 0;

    void* alignedPtr = pageAlignedBlockPtr(host_src, (NSUInteger)size_to_copy, &alignedLength);
    sourceOffset = uintptr_t(host_src) - uintptr_t(alignedPtr);
    id<MTLBuffer> sourceBuffer = [device newBufferWithBytesNoCopy:alignedPtr
                                                           length:alignedLength
                                                          options:options
                                                      deallocator:nil];

    uint64_t profile_id =
        getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst, size_to_copy, non_blocking);

    stream->copy_and_sync(
        sourceBuffer, destBuffer, size_to_copy, sourceOffset, dst_byte_offset, non_blocking, profile_id);
    [sourceBuffer release];
  }
}

static at::Tensor& copy_to_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  // Typecast to dst_ if needed and expand, which is a no-op
  Tensor src = (src_.dtype() != dst_.dtype() ? src_.to(dst_.dtype()) : src_).expand_as(dst_);

  // If src is not densely mapped in storage it must be cloned
  // It does not mean that tensor is contiguous, but rather
  // that it could be represented as 1d view
  if (!is_dense_in_storage(src)) {
    src = src.clone();
    TORCH_INTERNAL_ASSERT(is_dense_in_storage(src));
  }
  Tensor dst = dst_;
  bool needs_copy = false;
  // If src and dst_ strides do not match, it means that
  // either dst_ is not representable as 1d view or its stride order is different
  // in that case create an empty storage like src, copy it to device and then do
  // reshaping on the device
  if (src.strides() != dst_.strides()) {
    needs_copy = true;
    dst = at::empty_like(src, at::device(at::kMPS));
  }
  copy_to_mps_stride_contig(dst, src, non_blocking && !needs_copy);
  return needs_copy ? dst_.copy_(dst) : dst_;
}

void copy_blit_mps(void* dst, const void* src, size_t size) {
  // we don't have tensors info for profiling here
  uint64_t profile_id =
      getMPSProfiler().beginProfileCopy(src, dst, at::OptionalTensorRef(), at::OptionalTensorRef(), size, false);

  MPSStream* stream = getCurrentMPSStream();
  stream->copy_and_sync((id<MTLBuffer>)(src), (id<MTLBuffer>)(dst), size, 0, 0, true, profile_id);
}

static at::Tensor& copy_kernel_mps(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  auto src_byte_offset = src_.storage_offset() * src_.itemsize();
  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();

  // If dst is contiguous and there is no byte offset, we can save directly the result of
  // gather into dst. This reduces the overhead of doing an additional blit for most cases
  bool returnGatherOutput = dst_.is_contiguous();
  Tensor src;
  auto sameMemFormat =
      src_.is_contiguous(dst_.suggest_memory_format()) && dst_.is_contiguous(dst_.suggest_memory_format());
  const bool sameDataType = src_.dtype() == dst_.dtype();

  if ((!src_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) ||
      // the copy_cast path requires storage_offset to be applied before casting
      (src_.storage_offset() && !sameDataType)) {
    Tensor emptyShell = Tensor();
    src = gatherViewTensor(src_, returnGatherOutput ? dst_ : emptyShell);

    if (src.has_storage()) {
      if (returnGatherOutput) {
        return dst_;
      }

      src_byte_offset = 0;
    } else {
      src = src_.expand_as(dst_).contiguous();
      src_byte_offset = src.storage_offset() * src.itemsize();
    }
  } else {
    src = src_;
  }
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst_);
  id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);

  // Scatter to `dst` if the memory is not contiguous
  // If the memory is not contiguous, it means that the tensor has strides and we would not be
  // able to do the copy using a single blit
  if (!dst_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) {
    return scatterViewTensor(src, dst_);
  }
  src._set_conj(src_.is_conj());
  src._set_neg(src_.is_neg());

  MPSStream* stream = getCurrentMPSStream();
  if (sameDataType) {
    uint64_t profile_id = getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst_, src.nbytes(), true);
    // for GPU to GPU copies we only encode to stream's command buffer (no flushing)
    stream->copy(sourceBuffer, destBuffer, src.nbytes(), src_byte_offset, dst_byte_offset, profile_id);
  } else {
    if (dst_byte_offset) {
      auto tmp = at::empty(dst_.sizes(), dst_.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
      auto tmpBuffer = getMTLBufferStorage(tmp);
      copy_cast_mps(tmp, src, tmpBuffer, sourceBuffer);

      uint64_t profile_id = getMPSProfiler().beginProfileCopy(tmpBuffer, destBuffer, tmp, dst_, dst_.nbytes(), true);
      stream->copy(tmpBuffer, destBuffer, dst_.nbytes(), 0, dst_byte_offset, profile_id);
    } else {
      copy_cast_mps(dst_, src, destBuffer, sourceBuffer);
    }
  }
  return dst_;
}

at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  bool needs_broadcasting = false;

  if (src.numel() == 0 || dst.is_same(src)) {
    return dst;
  }
  if (dst.numel() == 0) {
    dst.resize_as_(src);
  }

  TORCH_CHECK(dst.dim() >= src.dim());
  if (dst.dim() > src.dim()) {
    needs_broadcasting = true;
  } else {
    const IntArrayRef src_sizes = src.sizes();
    const IntArrayRef dst_sizes = dst.sizes();
    for (const auto j : c10::irange(src.dim())) {
      if (src_sizes[j] == 1 && dst_sizes[j] != 1) {
        needs_broadcasting = true;
        break;
      }
    }
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kCPU) {
    return copy_from_mps_(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }
  if (src.device().type() == at::kCPU && dst.device().type() == at::kMPS) {
    return copy_to_mps_(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kMPS) {
    return copy_kernel_mps(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }
  TORCH_INTERNAL_ASSERT(src.device().type() == DeviceType::MPS, "mps_copy_ is implemented only for *->MPS; MPS->*");
  return dst;
}
} // namespace mps

Tensor _copy_from_and_resize_mps(const at::Tensor& self, const at::Tensor& dst) {
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, false);
}

Tensor _copy_from_mps(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, non_blocking);
}

} // namespace at::native
