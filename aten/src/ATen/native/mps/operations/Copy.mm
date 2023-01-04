//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <iostream>
#include <cstring>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <torch/library.h>
#include <ATen/native/Resize.h>
#include <c10/util/Optional.h>


namespace at {
namespace native {
namespace mps {

void* pageAlignedBlockPtr(
    const void* ptr,
    NSUInteger size,
    NSUInteger* alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  assert(address >= alignedAddress);
  assert(address + size <= alignedAddress + alignedLength);

  *alignedBlockSize = alignedLength;
  return (void*)alignedAddress;
}

/**
 * Computes number of elements one needs to transfer to preserve all the elements
 */
size_t compute_strided_size(const at::Tensor& t) {
   size_t rc = 1;
   if (t.numel() == 0) {
       return 0;
   }
   for(const auto i: c10::irange(t.dim())) {
     assert(t.size(i) > 0);
     rc += (t.size(i) - 1) * t.stride(i);
   }
   return rc;
}

bool is_strided_contiguous(const at::Tensor& t) {
  return compute_strided_size(t) == t.numel();
}

// Copy sourceBuffer into destBuffer, casting sourceBuffer to src.scalar_type().
// The shapes and dtypes are taken from dst and src, but their storage pointers are not used.
void copy_cast_mps(at::Tensor& dst, const at::Tensor& src,
                   id<MTLBuffer> destBuffer, id<MTLBuffer> sourceBuffer, bool non_blocking = true) {
  using namespace mps;

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSDataType dstDType = getMPSDataType(dst.scalar_type());
  MPSDataType srcDType = getMPSDataType(src.scalar_type());
  MPSShape* dstShape = getMPSShape(dst);
  MPSShape* srcShape = getMPSShape(src);

  @autoreleasepool {
    string key = "copy_cast_mps" + getTensorsStringKey({src, dst});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if (!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, src);
          MPSGraphTensor* outputTensor = [mpsGraph castTensor:inputTensor toType:dstDType name:@"cast"];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    MPSGraphTensorData* srcData = [[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:sourceBuffer shape:srcShape dataType:srcDType]
                                   autorelease];
    MPSGraphTensorData* dstData = [[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:destBuffer shape:dstShape dataType:dstDType]
                                   autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{cachedGraph->inputTensor_: srcData};
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{cachedGraph->outputTensor_: dstData};
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    if (!non_blocking)
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
  }
}

static at::Tensor& copy_from_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking)
{
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* stream = getCurrentMPSStream();
  Tensor dst;
  Tensor src;
  if (!dst_.is_contiguous()) {
    dst = at::empty_like(dst_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    dst = dst_;
  }

  auto storage_byte_offset = src_.storage_offset() * src_.itemsize();
  if (!src_.is_contiguous()) {
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
    MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;

    void* host_dst = dst.storage().data();
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
          tmp = at::native::empty_mps(dst.sizes(), dst.scalar_type(), c10::nullopt, kMPS);
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

      stream->copy_and_sync(tmpBuffer, destBuffer, size_to_copy, storage_byte_offset, destOffset, non_blocking);
      [destBuffer release];
    }
  }
  if (!dst.is_same(dst_)) {
    dst_.copy_(dst, non_blocking);
  }

  return dst_;
}

// Copies tensor from cpu to mps backed by identical strided-contiguous data
static void copy_to_mps_stride_contig(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
  MPSStream* stream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  auto dst_byte_offset = dst.storage_offset() * dst.itemsize();
  auto src_byte_offset = src.storage_offset() * src.itemsize();
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst);
  const size_t size_to_copy = src.nbytes();
  const void* host_src = static_cast<char *>(src.storage().data()) + src_byte_offset;

  TORCH_INTERNAL_ASSERT(src.dtype() == dst.dtype() && src.strides() == dst.strides() && is_strided_contiguous(src));

  @autoreleasepool {
    MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;
    NSUInteger sourceOffset = 0;

    void* alignedPtr = pageAlignedBlockPtr(host_src, (NSUInteger)size_to_copy, &alignedLength);
    id<MTLBuffer> sourceBuffer = [device newBufferWithBytesNoCopy:alignedPtr
                                          length:alignedLength
                                         options:options
                                     deallocator:nil];
    sourceOffset = uintptr_t(host_src) - uintptr_t(alignedPtr);

    stream->copy_and_sync(sourceBuffer, destBuffer, size_to_copy, sourceOffset, dst_byte_offset, non_blocking);
    [sourceBuffer release];
  }
}

static at::Tensor& copy_to_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking)
{
  // Typecast to dst_ if needed and expand, which is a no-op
  Tensor src = (src_.dtype() != dst_.dtype() ? src_.to(dst_.dtype()) : src_).expand_as(dst_);

  // If src is not contiguously strided it must be cloned
  // It does not mean that tensor is contiguous, but rather
  // that it could be represented as 1d view
  if (!is_strided_contiguous(src)) {
    src = src.clone();
    TORCH_INTERNAL_ASSERT(is_strided_contiguous(src));
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
  return needs_copy? dst_.copy_(dst) : dst_;
}

void copy_blit_mps(void* dst, const void* src, size_t size) {
  MPSStream* stream = getCurrentMPSStream();
  stream->copy_and_sync((id<MTLBuffer>)(src), (id<MTLBuffer>)(dst), size, 0, 0, true);
}

static at::Tensor& copy_kernel_mps(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking)
{
  auto src_byte_offset = src_.storage_offset() * src_.itemsize();
  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();

  // If dst is contiguous and there is no byte offset, we can save directly the result of
  // gather into dst. This reduces the overhead of doing an additional blit for most cases
  bool returnGatherOutput = (dst_.is_contiguous() && !dst_byte_offset);
  Tensor src;

  if (!src_.is_contiguous()) {
    Tensor emptyShell = Tensor();
    src = gatherViewTensor(src_, returnGatherOutput ? dst_ : emptyShell);

    if (src.has_storage()) {
      if (returnGatherOutput)
        return dst_;

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
  if (!dst_.is_contiguous()) {
    Tensor tmp;
    if (src.dtype() != dst_.dtype()) {
      id<MTLBuffer> tmpBuffer = sourceBuffer;
      if (src.element_size() < dst_.element_size()) {
        tmp = at::native::empty_mps(dst_.sizes(), dst_.scalar_type(), c10::nullopt, kMPS);
        tmpBuffer = getMTLBufferStorage(tmp);
      }

      copy_cast_mps(dst_, src, tmpBuffer, sourceBuffer);
    }

    return scatterViewTensor((src.dtype() != dst_.dtype() && tmp.has_storage()) ? tmp : src, dst_);
  }
  src._set_conj(src_.is_conj());
  src._set_neg(src_.is_neg());

  const size_t src_size = src.nbytes();
  if (src.dtype() == dst_.dtype()) {
    MPSStream* stream = getCurrentMPSStream();
    // for GPU to GPU copies we only encode to stream's command buffer (no flushing)
    stream->copy(sourceBuffer, destBuffer, src_size, src_byte_offset, dst_byte_offset);
  } else {
    copy_cast_mps(dst_, src, destBuffer, sourceBuffer);
  }
  return dst_;
}

at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking)
{
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  if (src.numel() == 0 || dst.is_same(src)) {
    return dst;
  }
  if (dst.numel() == 0) {
    dst.resize_as_(src);
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kCPU) {
    return copy_from_mps_(dst, src, non_blocking);
  }
  if (src.device().type() == at::kCPU && dst.device().type() == at::kMPS) {
    return copy_to_mps_(dst, src, non_blocking);
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kMPS) {
    return copy_kernel_mps(dst, src, non_blocking);
  }
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::MPS,
      "mps_copy_ is implemented only for *->MPS; MPS->*");
  return dst;
}
} // namespace mps

Tensor _copy_from_and_resize_mps(const at::Tensor& self, const at::Tensor& dst)
{
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, false);
}

Tensor _copy_from_mps(const at::Tensor& self, const at::Tensor& dst, bool non_blocking)
{
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, non_blocking);
}
} // namespace native
} // namespace at
