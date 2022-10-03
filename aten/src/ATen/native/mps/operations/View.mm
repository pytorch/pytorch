//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/mps/MPSAllocator.h>

namespace at {
namespace native {
namespace mps {

struct ViewCachedGraph : public MPSCachedGraph
{
  ViewCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor = nil;
  MPSGraphTensor* outputTensor = nil;
  MPSGraphTensor* updatesTensor = nil;
  MPSGraphTensor* storageOffsetTensor = nil;
  std::vector<MPSGraphTensor*> strideTensors;
};

static std::string getStridedKey(const ScalarType& dtype, const IntArrayRef& base_shape,
                          const IntArrayRef& new_shape, bool is_scatter)
{
  return (is_scatter ? "scatter:" : "gather:") + getMPSTypeString(dtype) + "[" +
         getArrayRefString(base_shape) + "]:[" + getArrayRefString(new_shape) + "]";
}

// initializes the MTLBuffers for tensor data and runs the MPSGraph for the view op
static Tensor& runViewGraph(ViewCachedGraph* cachedGraph, const at::Tensor& src, Tensor& output,
                            bool needsScatter, bool requires_sync = false)
{
  const id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  const id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);

  const IntArrayRef& strides   = needsScatter ? output.strides() : src.strides();
  const IntArrayRef& sizes     = needsScatter ? output.sizes() : src.sizes();
  const int64_t storage_offset = needsScatter ? output.storage_offset() : src.storage_offset();
  const MPSDataType inputType  = [cachedGraph->inputTensor dataType];

  MPSShape *inputShape = [cachedGraph->inputTensor shape];
  MPSShape *outputShape = needsScatter ? inputShape : getMPSShape(src);

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    // in case of scatter, we use output tensor as input buffer and write the results back to the source buffer
    feeds[cachedGraph->inputTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer: needsScatter ? outputBuffer : sourceBuffer
                                                                               shape: inputShape
                                                                            dataType: inputType] autorelease];
    if (needsScatter) {
      feeds[cachedGraph->updatesTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer: sourceBuffer
                                                                                   shape: getMPSShape(src.numel())
                                                                                dataType: inputType] autorelease];
    }
    MPSScalar storageOffsetScalar = getMPSScalar(storage_offset, ScalarType::Int);
    feeds[cachedGraph->storageOffsetTensor] = getMPSGraphTensorFromScalar(stream, storageOffsetScalar);

    std::vector<MPSScalar> strideScalars(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      strideScalars[i] = getMPSScalar(strides[i], ScalarType::Int);
      feeds[cachedGraph->strideTensors[i]] = getMPSGraphTensorFromScalar(stream, strideScalars[i]);
    }
    // Workaround for MPSShaderLibrary bug
    // TODO: Remove once https://github.com/pytorch/pytorch/issues/82305 is resolved
    auto outputType = getMPSDataType(output.scalar_type());
    if (outputType ==  MPSDataTypeUInt8) {
        outputType =  MPSDataTypeInt8;
    }
    MPSGraphTensorData* outputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer: outputBuffer
                                                                                    shape: outputShape
                                                                                 dataType: outputType] autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cachedGraph->outputTensor : outputTensorData
    };
    stream->executeMPSGraph(cachedGraph->graph(), feeds, results,
                            requires_sync ? SyncType::COMMIT : SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

static MPSGraphTensor* chainViewOperation(ViewCachedGraph* cachedGraph, const IntArrayRef& size,
                                          const IntArrayRef& stride, int64_t offset,
                                          const IntArrayRef& base_shape, bool needsScatter,
                                          const bool needsBoolCast)
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
    NSData* shapeData = [NSData dataWithBytes: sizeArray.data()
                                       length: shape_size * sizeof(int32_t)];
    MPSGraphTensor* shapeTensor = [mpsGraph constantWithData: shapeData
                                                       shape: @[[NSNumber numberWithUnsignedInteger: shape_size]]
                                                    dataType: MPSDataTypeInt32];
    MPSGraphTensor* indicesTensor = nil;
    // create stride Tensors for each rank of the input tensor
    for (int i = 0; i < shape_size; i++) {
      MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis: (-i - 1)
                                                  withShapeTensor: shapeTensor
                                                             name: nil];
      MPSGraphTensor* strideTensor = cachedGraph->strideTensors[shape_size - i - 1];
      MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor: rangeTensor
                                                              secondaryTensor: strideTensor
                                                                         name: nil];
      if (!indicesTensor) {
        indicesTensor = indexTensor;
      } else {
        indicesTensor = [mpsGraph additionWithPrimaryTensor: indexTensor
                                            secondaryTensor: indicesTensor
                                                       name: nil];
      }
    }

    indicesTensor = [mpsGraph additionWithPrimaryTensor: indicesTensor
                                        secondaryTensor: cachedGraph->storageOffsetTensor
                                                   name: nil];
    MPSGraphTensor *inputTensor = cachedGraph->inputTensor;

    // Workaround for bool scatter/gather deficiency
    // See https://github.com/pytorch/pytorch/issues/82663
    if (needsBoolCast) {
      inputTensor = [mpsGraph castTensor:inputTensor
                                  toType:MPSDataTypeInt8
                                    name:@"Cast away from bool"];
    }

    MPSGraphTensor *reshapedInputTensor = [mpsGraph reshapeTensor: inputTensor
                                                        withShape: @[@-1]
                                                             name: nil];
    MPSGraphTensor *reshapedIndicesTensor = [mpsGraph reshapeTensor: indicesTensor
                                                          withShape: @[@-1]
                                                               name: nil];
    if (needsScatter) {
      MPSGraphTensor* scatteredTensor = [mpsGraph scatterAlongAxis: (NSInteger) 0
                                                    withDataTensor: reshapedInputTensor
                                                     updatesTensor: cachedGraph->updatesTensor
                                                     indicesTensor: reshapedIndicesTensor
                                                              mode: MPSGraphScatterModeSet
                                                              name: nil];
      outputTensor = [mpsGraph reshapeTensor: scatteredTensor
                                   withShape: getMPSShape(base_shape)
                                        name: nil];
    } else {
      // Call gather to coalesce the needed values. Result will be of same shape as flattened indices tensor
      MPSGraphTensor *gatheredTensor = [mpsGraph gatherWithUpdatesTensor: reshapedInputTensor
                                                           indicesTensor: reshapedIndicesTensor
                                                                    axis: 0
                                                         batchDimensions: 0
                                                                    name: nil];
      // Reshape the data to desired size
      outputTensor =  [mpsGraph reshapeTensor: gatheredTensor
                              withShapeTensor: shapeTensor
                                         name: nil];
    }

    // Workaround for bool scatter/gather deficiency
    // See https://github.com/pytorch/pytorch/issues/82663
    if (needsBoolCast) {
      outputTensor = [mpsGraph castTensor:outputTensor
                                   toType:MPSDataTypeBool
                                     name:@"Cast back to bool"];
    }
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
static ViewCachedGraph* createViewGraph(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t storage_offset, bool needsScatter)
{
  IntArrayRef base_shape = get_buffer_shape(self.storage().data());
  if (base_shape.size() == 0) {
    // IntArrayRef wouldn't own the data, so we use a static storage
    static const int64_t shape_1d = 1;
    // self.sizes().size() could be zero
    base_shape = self.sizes().size() ? self.sizes() :
                      self.is_view() ? self._base().sizes() : IntArrayRef(&shape_1d, 1);

    // base_shape will be retained in MPSAllocator until buffer gets recycled
    if (self.storage().data())
      set_buffer_shape(self.storage().data(), base_shape);
  }
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = getStridedKey(self.scalar_type(), base_shape, size, needsScatter);
    ViewCachedGraph* cachedGraph = static_cast<ViewCachedGraph *>(cache_->LookUp(key));

    if (!cachedGraph) {
      cachedGraph = static_cast<ViewCachedGraph *>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        ViewCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new ViewCachedGraph(mpsGraph);
            // Workaround for MPSShaderLibrary bug
            // TODO: Remove once https://github.com/pytorch/pytorch/issues/82305 is resolved
            auto inputType = getMPSScalarType(self.scalar_type());
            if (inputType ==  MPSDataTypeUInt8) {
                inputType =  MPSDataTypeInt8;
            }
            auto needsBoolCast = inputType == MPSDataTypeBool;
            // Self is the input tensor we are creating view of
            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(base_shape));
            newCachedGraph->storageOffsetTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]);
            for (int i = 0; i < size.size(); i++) {
              newCachedGraph->strideTensors.push_back(mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]));
            }
            if (needsScatter) {
              newCachedGraph->updatesTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, inputType);
            }
            newCachedGraph->outputTensor = chainViewOperation(newCachedGraph, size, stride, storage_offset, base_shape, needsScatter, needsBoolCast);
        }
        return newCachedGraph;
      }));
    }
    return cachedGraph;
  }
}

Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst)
{
  ViewCachedGraph* cachedGraph = nullptr;

  const IntArrayRef& base_shape = get_buffer_shape(src.storage().data());
  if (base_shape.size() > 0) {
    string key = getStridedKey(src.scalar_type(), base_shape, src.sizes(), /*is_scatter*/ false);
    cachedGraph = static_cast<ViewCachedGraph *>(MPSGraphCache::getInstance()->LookUp(key));
  }
  // there are cases where gatherViewTensor() is called without having as_strided() called beforehand.
  // this typically may come from copy_mps variants. In such cases, when the base_shape isn't found the
  // callers would resort to make the tensor contiguous in an alternative code path.
  if (!cachedGraph) {
    return Tensor();
  }

  bool requires_sync = false;
  Tensor output;
  if (!dst.has_storage()) {
    output = at::native::empty_mps(src.sizes(), src.scalar_type(), c10::nullopt, kMPS);
    requires_sync = true;
  }
  return runViewGraph(cachedGraph, src, dst.has_storage() ? dst : output, /*needsScatter*/ false, requires_sync);
}

Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output)
{
  ViewCachedGraph* cachedGraph = createViewGraph(output, output.sizes(), output.strides(),
                                                 output.storage_offset(), /*needsScatter*/ true);
  return runViewGraph(cachedGraph, src, output, /*needsScatter*/ true, /*requires_sync*/  true);
}

} // namespace mps

// implementation of as_strided() op
Tensor as_strided_tensorimpl_mps(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_)
{
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // 0 sizes won't result in any change in the shape of the Tensor so we can skip it.
  if (size.size() > 0)
    mps::createViewGraph(self, size, stride, storage_offset, /*needsScatter*/ false);

  return result;
}

} // namespace native
} // namespace at
