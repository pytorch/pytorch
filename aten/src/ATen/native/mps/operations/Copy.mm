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

MPSGraphTensor* chainViewOperation(MPSGraph* mpsGraph, IntArrayRef size,
                             IntArrayRef stride, int64_t storage_offset,
                             MPSGraphTensor* inputTensor, const Tensor& self) {
  MPSGraphTensor *outputTensor = nil;
  const size_t shape_size = size.size();

  @autoreleasepool {
      std::vector<int32_t> sizeArray(shape_size);
      const int64_t int_max = std::numeric_limits<int32_t>::max();
      for (int i = 0; i < shape_size; i++) {
        TORCH_CHECK(size[i] <= int_max);
        sizeArray[i] = static_cast<int32_t>(size[i]);
      }
      NSData* shapeData = [NSData dataWithBytes:sizeArray.data()
                                         length:shape_size * sizeof(int32_t)];
      MPSGraphTensor* shapeTensor =  [mpsGraph constantWithData:shapeData
                                                          shape:@[[NSNumber numberWithUnsignedInteger: shape_size]]
                                                       dataType:MPSDataTypeInt32];

      MPSGraphTensor* storageOffsetTensor = [mpsGraph constantWithScalar:storage_offset
                                                                dataType:MPSDataTypeInt32];
      MPSGraphTensor* strideTensor = [mpsGraph constantWithScalar:stride[shape_size - 1]
                                                         dataType:MPSDataTypeInt32];
      MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis:-1
                                                  withShapeTensor:shapeTensor
                                                             name:nil];
      MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor:rangeTensor
                                                              secondaryTensor:strideTensor
                                                                         name:nil];
      MPSGraphTensor* indicesTensor = indexTensor;
      // create stride Tensors for each rank of the input tensor
      for (int i = 1; i < shape_size; i++) {
        strideTensor = [mpsGraph constantWithScalar:stride[shape_size - i - 1]
                                           dataType:MPSDataTypeInt32];
        MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis:(-i - 1)
                                                    withShapeTensor:shapeTensor
                                                               name:nil];
        MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor:rangeTensor
                                                                secondaryTensor:strideTensor
                                                                           name:nil];
        indicesTensor = [mpsGraph additionWithPrimaryTensor:indexTensor
                                            secondaryTensor:indicesTensor
                                                       name:nil];
      }
      indicesTensor = [mpsGraph additionWithPrimaryTensor:indicesTensor
                                          secondaryTensor:storageOffsetTensor
                                                     name:nil];
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
  if (size.size() > 0) {

    // If self itself was a view tensor, that means we need to chain the graphs
    // else we will create a new entry in the cache
    struct CachedGraph : public MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* inputTensor_ = nil;
      MPSGraphTensor* outputTensor_ = nil;
    };

    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    @autoreleasepool {
      string key = mps::getStridedKey(self, size, stride, storage_offset);
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if (!cachedGraph) {
        // Check if this stride operation is already performed on a strided tensor
        auto origKey = getStridedKey(self, self.sizes(), self.strides(), self.storage_offset());
        auto origGraph = static_cast<CachedGraph *>(cache_->LookUp(origKey));
        cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
          CachedGraph *newCachedGraph = nil;
          @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);

              // Self is the input tensor we are creating view of, which can also be a stride
              // In later case, preserve shape from the original graph
              auto shape = origGraph ? [origGraph->inputTensor_ shape] : getMPSShape(self);
              auto inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self.scalar_type()), shape);
              newCachedGraph->inputTensor_ = inputTensor;
              newCachedGraph->outputTensor_ = chainViewOperation(mpsGraph, size, stride,
                                                                 storage_offset, inputTensor, self);
          }
          return newCachedGraph;
        }, self.storage().data());
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

static bool copy_requires_temporaries(const Tensor& dst, const Tensor& src) {
  bool same_dtype = src.dtype() == dst.dtype();
  if (same_dtype && src.is_contiguous() && dst.is_contiguous()) {
    return false;
  } else {
    return true;
  }
}

// Copy sourceBuffer into destBuffer, casting sourceBuffer to src.scalar_type().
// The shapes and dtypes are taken from dst and src, but their storage pointers are not used.
void copy_cast_mps(at::Tensor& dst, const at::Tensor& src,
                   id<MTLBuffer> destBuffer, id<MTLBuffer> sourceBuffer) {
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
  }
}

static at::Tensor& copy_from_mps_(at::Tensor& dst_, const at::Tensor& src_,
                           bool non_blocking) {

  using namespace mps;
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* stream = getCurrentMPSStream();
  uint64_t size = src_.nbytes();
  if (size == 0) return dst_;
  Tensor dst;
  Tensor src;
  if (!dst_.is_contiguous()) {
    dst = at::empty_like(dst_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    dst = dst_;
  }

  auto storage_byte_offset = src_.storage_offset() * src_.itemsize();
  id<MTLBuffer> sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src_.storage().data());
  if (!src_.is_contiguous()) {
    id<MTLBuffer> gatherTensor = gatherViewTensor(src_, sourceBuffer);
    if (gatherTensor) {
      sourceBuffer = gatherTensor;
      storage_byte_offset = 0;
    } else {
      src = src_.expand_as(dst).contiguous();
      sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src.storage().data());
      storage_byte_offset = src.storage_offset() * src.itemsize();
    }
  } else {
    src = src_;
  }

  void* host_dst = dst.storage().data();

  if (sourceBuffer == nil) return dst_;
  NSUInteger destOffset = dst.storage_offset() * dst.itemsize();

  // In case of dtype change, first convert src inplace
  if (src_.dtype() != dst_.dtype()) {
    copy_cast_mps(dst_, src_, sourceBuffer, sourceBuffer);
  }

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
  if (!dst.is_same(dst_)) {
    dst_.copy_(dst, non_blocking);
  }

  return dst_;
}

static at::Tensor& copy_to_mps_(at::Tensor& dst_, const at::Tensor& src_,
                         bool non_blocking) {
  MPSStream* stream = getCurrentMPSStream();
  Tensor dst;
  Tensor src;

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();
  id<MTLBuffer> destBuffer = __builtin_bit_cast(id<MTLBuffer>, dst_.storage().data());


  if (src_.is_view()) {
    src = src_.to(dst_.dtype()).expand_as(dst_);
    if (dst_.is_contiguous()) {
      src = src.contiguous();
    }
  } else {
    src = src_;
    if (src.dtype() != dst_.dtype()) {
      // In case of dtype change, perform conversion on source device
      src = src.to(dst_.dtype());
    }
  }

  const void* host_src = src.storage().data();
  uint64_t size = src.nbytes();

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
    if (src_.is_view() || !src_.is_contiguous())
      sourceOffset += src_.storage_offset() * src_.itemsize();

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

  return dst_;
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


static at::Tensor& copy_kernel_mps(at::Tensor& dst_, const at::Tensor& src_,
                            bool non_blocking) {
  uint64_t size = src_.nbytes();
  auto src_byte_offset = src_.storage_offset() * src_.itemsize();
  id<MTLBuffer> sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src_.storage().data());
  Tensor src;
  if (!src_.is_contiguous()) {
    id<MTLBuffer> gatherTensor = gatherViewTensor(src_, sourceBuffer);
    if (gatherTensor) {
      sourceBuffer = gatherTensor;
      src_byte_offset = 0;
    } else {
      src = src_.expand_as(dst_).contiguous();
      sourceBuffer = __builtin_bit_cast(id<MTLBuffer>, src.storage().data());
      src_byte_offset = src.storage_offset() * src.itemsize();
    }
  } else {
    src = src_;
  }
  src._set_conj(src_.is_conj());
  src._set_neg(src_.is_neg());

  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();
  id<MTLBuffer> destBuffer = __builtin_bit_cast(id<MTLBuffer>, dst_.storage().data());

  if (src.dtype() == dst_.dtype()) {
    MPSStream* stream = getCurrentMPSStream();
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
        stream->commitAndWait();
      }
    });
  } else {
    copy_cast_mps(dst_, src_, destBuffer, sourceBuffer);
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
