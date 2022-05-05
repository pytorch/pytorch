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
#include <torch/library.h>

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

static at::Tensor& copy_from_mps_(at::Tensor& self, const at::Tensor& src,
                           bool non_blocking) {

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* stream = getCurrentMPSStream();
  uint64_t size = src.nbytes();
  if (size == 0) return self;
  void* host_dst = self.data_ptr();

  // MTLContext* context = static_cast<MTLContext *>(device->device_handle);
  auto storage_byte_offset = src.storage_offset() * src.itemsize();
  id<MTLBuffer> sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src.storage().data());
  if (sourceBuffer == nil) return self;
  NSUInteger destOffset = 0;

  @autoreleasepool {
    MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;

    void* alignedPtr = pageAlignedBlockPtr(host_dst, (NSUInteger)size, &alignedLength);
    id<MTLBuffer> destBuffer = [device newBufferWithBytesNoCopy:alignedPtr
                                                         length:alignedLength
                                                        options:options
                                                    deallocator:nil];
    destOffset = uintptr_t(host_dst) - uintptr_t(alignedPtr);
    // 4 bytes alignment required on macos for blits.
    TORCH_CHECK(destOffset % 4 == 0, "Unaligned blit request");

    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
        id<MTLBlitCommandEncoder> blitEncoder =
            [commandBuffer blitCommandEncoder];

        [blitEncoder copyFromBuffer:sourceBuffer
                       sourceOffset:(NSUInteger)storage_byte_offset
                           toBuffer:destBuffer
                  destinationOffset:(NSUInteger)destOffset
                               size:(NSUInteger)size];
        [blitEncoder endEncoding];

        if (non_blocking) {
          stream->commit(true);
        } else {
          stream->commitAndWait();
        }
        [destBuffer release];
      }
    });
  }

  return self;
}

static at::Tensor& copy_to_mps_(at::Tensor& self, const at::Tensor& src,
                         bool non_blocking) {
  MPSStream* stream = getCurrentMPSStream();
  const void* host_src = src.data_ptr();
  uint64_t size = src.nbytes();

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  auto dst_byte_offset = self.storage_offset() * self.itemsize();
  id<MTLBuffer> destBuffer = __builtin_bit_cast(id<MTLBuffer>, self.storage().data());

  NSUInteger sourceOffset = 0;
  @autoreleasepool {
    MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;

    void* alignedPtr = pageAlignedBlockPtr(host_src, (NSUInteger)size, &alignedLength);
    id<MTLBuffer> sourceBuffer = [device newBufferWithBytesNoCopy:alignedPtr
                                          length:alignedLength
                                         options:options
                                     deallocator:nil];
    sourceOffset = uintptr_t(host_src) - uintptr_t(alignedPtr);

    dispatch_sync(stream->queue(), ^() {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
        id<MTLBlitCommandEncoder> blitEncoder =
            [commandBuffer blitCommandEncoder];

        [blitEncoder copyFromBuffer:sourceBuffer
                       sourceOffset:(NSUInteger)sourceOffset
                           toBuffer:destBuffer
                  destinationOffset:(NSUInteger)dst_byte_offset
                               size:(NSUInteger)size];
        [blitEncoder endEncoding];
        if (non_blocking) {
          stream->commit(true);
        } else {
          stream->commitAndWait();
        }
      }
    });
    [sourceBuffer release];
  }

  return self;
}

void copy_blit_mps(void* dst, const void* src, size_t size) {
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> sourceBuffer = (id<MTLBuffer>)(src);
  id<MTLBuffer> destBuffer = (id<MTLBuffer>)(dst);
  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
      id<MTLBlitCommandEncoder> blitEncoder =
          [commandBuffer blitCommandEncoder];

      [blitEncoder copyFromBuffer:sourceBuffer
                     sourceOffset:0
                         toBuffer:destBuffer
                destinationOffset:0
                             size:size];
      [blitEncoder endEncoding];
      stream->commitAndWait();
    }
  });
}


static at::Tensor& copy_kernel_mps(at::Tensor& dst, const at::Tensor& src,
                            bool non_blocking) {
  MPSStream* stream = getCurrentMPSStream();
  uint64_t size = src.nbytes();

  auto src_byte_offset = src.storage_offset() * src.itemsize();
  id<MTLBuffer> sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src.storage().data());

  auto dst_byte_offset = dst.storage_offset() * dst.itemsize();
  id<MTLBuffer> destBuffer = __builtin_bit_cast(id<MTLBuffer>, dst.storage().data());

  dispatch_sync(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

      [blitEncoder copyFromBuffer:sourceBuffer
                     sourceOffset:src_byte_offset
                         toBuffer:destBuffer
                destinationOffset:dst_byte_offset
                             size:size];
      [blitEncoder endEncoding];
      if (non_blocking) {
        stream->commit(true);
      } else {
        stream->commitAndWait();
      }
    }
  });
  return dst;
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

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst)
{
  return mps_copy_(const_cast<Tensor&>(dst), self, false);
}

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking)
{
  return mps_copy_(const_cast<Tensor&>(dst), self, non_blocking);
}

} // namespace mps

TORCH_LIBRARY_IMPL(aten, MPS, m) {
   m.impl("_copy_from", TORCH_FN(mps::_copy_from));
   m.impl("_copy_from_and_resize", TORCH_FN(mps::_copy_from_and_resize));
}

} // namespace native
} // namespace at
