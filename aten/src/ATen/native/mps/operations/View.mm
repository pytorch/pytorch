//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>

namespace at {

// these are from MPSAllocator
namespace mps {
  // to check the requeted non-aligned size of an MTL buffer
  ssize_t get_requested_buffer_size(void* ptr);
  // to retrieve the shape of a base tensor from a view tensor
  IntArrayRef get_buffer_shape(void* ptr);
  // to set the shape of a base tensor from a view tensor
  void set_buffer_shape(void* ptr, const IntArrayRef& shape);
}

namespace native {
namespace mps {

struct ViewCachedGraph : public MPSCachedGraph
{
  ViewCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor = nil;
  MPSGraphTensor* outputTensor = nil;
  MPSGraphTensor* storageOffsetTensor = nil;
  std::vector<MPSGraphTensor*> strideTensors;
};

std::string getStridedKey(const Tensor& self, const IntArrayRef& base_shape, const IntArrayRef& new_shape)
{
  return "as_strided:" +
         getMPSTypeString(self.scalar_type()) + "[" +
         mps::getArrayRefString(base_shape) + "]:[" +
         mps::getArrayRefString(new_shape) + "]";
}

MPSGraphTensor* chainViewOperation(ViewCachedGraph* cachedGraph, const IntArrayRef& size,
                                   const IntArrayRef& stride, int64_t offset)
{
  MPSGraph* mpsGraph = cachedGraph->graph();
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
      MPSGraphTensor* shapeTensor = [mpsGraph constantWithData:shapeData
                                                         shape:@[[NSNumber numberWithUnsignedInteger: shape_size]]
                                                      dataType:MPSDataTypeInt32];
      MPSGraphTensor* indicesTensor = nil;
      // create stride Tensors for each rank of the input tensor
      for (int i = 0; i < shape_size; i++) {
        MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis:(-i - 1)
                                                    withShapeTensor:shapeTensor
                                                               name:nil];
        MPSGraphTensor* strideTensor = cachedGraph->strideTensors[shape_size - i - 1];
        MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor:rangeTensor
                                                                secondaryTensor:strideTensor
                                                                           name:nil];
        if (!indicesTensor) {
          indicesTensor = indexTensor;
        } else {
          indicesTensor = [mpsGraph additionWithPrimaryTensor:indexTensor
                                              secondaryTensor:indicesTensor
                                                        name:nil];
        }
      }

      indicesTensor = [mpsGraph additionWithPrimaryTensor:indicesTensor
                                          secondaryTensor:cachedGraph->storageOffsetTensor
                                                     name:nil];
      MPSGraphTensor *reshapedInputTensor = [mpsGraph reshapeTensor:cachedGraph->inputTensor
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
Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t storage_offset)
{
  auto result = detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()),
                                                self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // 0 sizes won't result in any change in the shape of the Tensor so we can skip it.
  if (size.size() == 0)
    return result;

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    IntArrayRef base_shape = get_buffer_shape(self.storage().data());
    if (base_shape.size() == 0) {
      // self.sizes().size() could be zero
      base_shape = self.sizes().size() ? self.sizes() : IntArrayRef({1});
      // base_shape will be retained in MPSAllocator until buffer gets recycled
      if (self.storage().data())
        set_buffer_shape(self.storage().data(), base_shape);
    }

    string key = mps::getStridedKey(self, base_shape, size);
    ViewCachedGraph* cachedGraph = static_cast<ViewCachedGraph *>(cache_->LookUp(key));
    if (!cachedGraph) {
      cachedGraph = static_cast<ViewCachedGraph *>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        ViewCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new ViewCachedGraph(mpsGraph);
            // Self is the input tensor we are creating view of
            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(self.scalar_type()), getMPSShape(base_shape));
            newCachedGraph->storageOffsetTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]);
            for (int i = 0; i < size.size(); i++) {
              newCachedGraph->strideTensors.push_back(mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]));
            }
            newCachedGraph->outputTensor = chainViewOperation(newCachedGraph, size, stride, storage_offset);
        }
        return newCachedGraph;
      }));
    }
  }
  return result;
}

Tensor gatherViewTensor(const at::Tensor& src)
{
  const id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  const IntArrayRef& base_shape = get_buffer_shape(sourceBuffer);
  // there are cases where gatherViewTensor() is called without having as_strided() called beforehand.
  // this typically may come from copy_mps variants. In such cases, when the base_shape isn't found the
  // callers would resort to make the tensor contiguous in an alternative code path.
  if (base_shape.size() == 0)
    return Tensor();

  const IntArrayRef& strides = src.strides();
  const IntArrayRef& sizes = src.sizes();
  const int64_t storage_offset = src.storage_offset();

  at::mps::MPSStream* stream = getCurrentMPSStream();

  string key = getStridedKey(src, base_shape, sizes);
  ViewCachedGraph* cachedGraph = static_cast<ViewCachedGraph *>(MPSGraphCache::getInstance()->LookUp(key));
  if (!cachedGraph) {
    return Tensor();
  }
  Tensor output = at::native::empty_mps(src.sizes(), src.scalar_type(), c10::nullopt, kMPS, c10::nullopt, MemoryFormat::Contiguous);

  @autoreleasepool {

    MPSGraphTensorData* inputTensorData =  [[[MPSGraphTensorData alloc] initWithMTLBuffer: sourceBuffer
                                                                                    shape: [cachedGraph->inputTensor shape]
                                                                                 dataType: [cachedGraph->inputTensor dataType]] autorelease];
    MPSGraphTensorData* outputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer: getMTLBufferStorage(output)
                                                                                    shape: getMPSShape(src.sizes())
                                                                                 dataType: getMPSDataType(src.scalar_type())] autorelease];
    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    feeds[cachedGraph->inputTensor] = inputTensorData;
    feeds[cachedGraph->storageOffsetTensor] = getMPSGraphTensorFromScalar(stream, Scalar(storage_offset), MPSDataTypeInt32);
    for (int i = 0; i < sizes.size(); i++) {
      feeds[cachedGraph->strideTensors[i]] = getMPSGraphTensorFromScalar(stream, Scalar(strides[i]), MPSDataTypeInt32);
    }
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cachedGraph->outputTensor : outputTensorData
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
  return output;
}

} // namespace mps

Tensor as_strided_tensorimpl_mps(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_)
{
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  return mps::as_strided_tensorimpl(self, size, stride, storage_offset);
}

} // namespace native
} // namespace at
