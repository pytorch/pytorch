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

MPSGraphTensor* chainViewOperation(MPSGraph* mpsGraph, IntArrayRef size,
                             IntArrayRef stride, int64_t storage_offset,
                             MPSGraphTensor* inputTensor, const Tensor& self) {
  MPSGraphTensor *outputTensor = nil;
  @autoreleasepool {
      int32_t* sizeArray = new int32_t[size.size()]();
      for (int i = 0; i < size.size(); i++) {
        sizeArray[i] = size[i];
      }
      NSData* shapeData = [NSData dataWithBytes : sizeArray
                                    length : size.size()*sizeof(int32_t)];

      //NSString *strData = [[NSString alloc]initWithData:shapeData encoding:NSUTF8StringEncoding];
      //NSLog(@"%@",strData);
      MPSGraphTensor* shapeTensor =  [mpsGraph constantWithData : shapeData
                                                        shape : @[[NSNumber numberWithUnsignedInteger: size.size()]]
                                                      dataType : MPSDataTypeInt32];
      MPSGraphTensor* storageOffsetTensor = [mpsGraph constantWithScalar :  storage_offset
                                                            dataType : MPSDataTypeInt32];
      MPSGraphTensor* strideTensor = [mpsGraph constantWithScalar : stride[self.dim()-1]
                                                      dataType : MPSDataTypeInt32];
      MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis:-1
                                                      withShapeTensor : shapeTensor
                                                              name : nil];
      MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor :  rangeTensor
                                                          secondaryTensor : strideTensor
                                                              name : nil];
      MPSGraphTensor* indicesTensor = indexTensor;
      // create stride Tensors for each rank of the input tensor
      for (int i = 1; i < self.dim(); i++) {
        strideTensor = [mpsGraph constantWithScalar : stride[self.dim() - i - 1]
                                        dataType : MPSDataTypeInt32];
        MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis: (-i-1)
                                                        withShapeTensor : shapeTensor
                                                                name : nil];
        MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor :  rangeTensor
                                                            secondaryTensor : strideTensor
                                                                name : nil];
        indicesTensor = [mpsGraph additionWithPrimaryTensor : indexTensor
                                          secondaryTensor : indicesTensor
                                              name : nil];
      }
      indicesTensor = [mpsGraph additionWithPrimaryTensor : indicesTensor
                                            secondaryTensor : storageOffsetTensor
                                                  name : nil];
      MPSGraphTensor *reshapedInputTensor = [mpsGraph reshapeTensor:inputTensor
                                                         withShape:@[@-1]
                                                              name:nil];
      MPSGraphTensor *reshapedIndicesTensor = [mpsGraph reshapeTensor:indicesTensor
                                                 withShape:@[@-1]
                                                      name:nil];
      // Call gather to coalesce the needed values. Result will be of same shape as flattened indices tensor
      MPSGraphTensor *gatheredTensor = [mpsGraph gatherWithUpdatesTensor:reshapedInputTensor
                                                        indicesTensor:reshapedIndicesTensor
                                                                 axis:0
                                                      batchDimensions:0
                                                                 name:nil];

      delete[] sizeArray;
      // Reshape the data to desired size
      outputTensor =  [mpsGraph reshapeTensor:gatheredTensor
                               withShapeTensor:shapeTensor
                                          name:nil];
  }
  return outputTensor;
}


// There are few cases we need to consider:
// Here nodes are the Tensors and the edges are the operations performed on the
// Tensor. As a result of the operation performed we can have result as View
// Tensor (View T) or a Non view tensor (NonView T). The difference is if its
// mapped by the same underlying storage ptr or a new MTLBuffer was allocated.
//                T = Tensor
//                 ----------
//                 | Orig T |
//                 ----------
//                /     |     \
//             View T  View T  NonView T
//             /      /    \      |
//            View T /      \     |
//            |     /        \    |
//            |    /          \   |
//            |   /            \  |
//            NonView T         NonView T
//
//
Tensor as_strided_tensorimpl_mps(const Tensor& self, IntArrayRef size,
                                 IntArrayRef stride,
                                 optional<int64_t> storage_offset_) {
  using namespace mps;
  // Use the size and stride to create a unique key
  auto result = detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  setStrided(result, size, stride, storage_offset);

  // 0 sizes won't result in any change in the shape of the Tensor so we can
  // skip it. Also if the memory is contiguous we don't need to do
  // gather-scatter operations using graph.
  if (size.size() > 0 && (!result.is_contiguous())) {

    // If self itself was a view tensor, that means we need to chain the graphs
    // else we will create a new entry in the cache
    struct CachedGraph : public MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* inputTensor_ = nil;
      MPSGraphTensor* outputTensor_ = nil;
      IntArrayRef size_;
      IntArrayRef stride_;
      int64_t storage_offset_;
    };

    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    @autoreleasepool {
      string lookup_key = mps::getStridedKey(self, self.sizes(), self.strides(),
                      self.storage_offset());
#if _DEBUG
      std::cout << "Lookup key " << lookup_key << std::endl;
#endif
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(lookup_key));

      if(!cachedGraph) {
        string insert_key = mps::getStridedKey(self,size, stride, storage_offset);
#if _DEBUG
        std::cout << "Insert key " << insert_key << std::endl;
#endif
        CachedGraph* insertCachedGraph = static_cast<CachedGraph *>(cache_->LookUp(insert_key));
        if (!insertCachedGraph) {
          MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(insert_key, ^ MPSCachedGraph * () {
            CachedGraph *newCachedGraph = nil;
            @autoreleasepool {
                MPSGraph* mpsGraph = make_mps_graph();
                newCachedGraph = new CachedGraph(mpsGraph);

                // Self is the input tensor we are creating view of
                MPSGraphTensor* inputTensor = [mpsGraph placeholderWithShape : getMPSShape(self)
                                                                    dataType : getMPSDataType(self.scalar_type())
                                                                    name : nil];
                newCachedGraph->inputTensor_ = inputTensor;
                newCachedGraph->outputTensor_ = chainViewOperation(mpsGraph, size,
                                                                   stride,
                                                                   storage_offset,
                                                                   inputTensor,
                                                                   self);
                newCachedGraph->size_ = size;
                newCachedGraph->stride_ = stride;
                newCachedGraph->storage_offset_ = storage_offset;
            }
            return newCachedGraph;
          });
          cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }
      } else {
        // Else part takes care of the chaining where multiple view operations
        // were implemented on the same underlying data storage ptr
        cachedGraph->outputTensor_ = chainViewOperation(cachedGraph->graph(),
                                      size, stride, storage_offset,
                                      cachedGraph->outputTensor_, self);
        cachedGraph->size_ = size;
        cachedGraph->stride_ = stride;
        cachedGraph->storage_offset_ = storage_offset;
      }
    }
  }
  return result;
}

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

  if (!src.is_contiguous()) {
    id<MTLBuffer> gatherTensor = gatherViewTensor(src, sourceBuffer);
    if (gatherTensor) {
      sourceBuffer = gatherTensor;
      storage_byte_offset = 0;
    }
  }

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
