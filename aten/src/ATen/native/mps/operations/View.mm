//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/mps/MPSAllocator.h>

namespace at::native {
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

static std::string getStridedKey(const ScalarType& self_dtype, const ScalarType& updates_dtype, const IntArrayRef& base_shape,
                                 const IntArrayRef& new_shape, const IntArrayRef& stride,
                                 int64_t storage_offset, bool is_scatter)
{
  std::string dtype_key = getMPSTypeString(self_dtype);
  if (is_scatter) {
    dtype_key += ":" + getMPSTypeString(updates_dtype);
  }

  return (is_scatter ? "scatter:" : "gather:") + dtype_key + "[" +
         getArrayRefString(base_shape) + "]:[" + getArrayRefString(new_shape) + "]:[" +
         getArrayRefString(stride) + "]:[" + to_string(storage_offset) + "]";
}

// initializes the MTLBuffers for tensor data and runs the MPSGraph for the view op
static Tensor& runViewGraph(ViewCachedGraph* cachedGraph, const at::Tensor& src, Tensor& output,
                            bool needsScatter, bool requires_sync = false) {
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
      auto updatesType = getMPSScalarType(src.scalar_type());
      if (updatesType == MPSDataTypeUInt8 || updatesType == MPSDataTypeBool) {
        updatesType = MPSDataTypeInt8;
      }

      feeds[cachedGraph->updatesTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer: sourceBuffer
                                                                                   shape: getMPSShape(src.numel())
                                                                                dataType: updatesType] autorelease];
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
MPSGraphTensorData* getMPSGraphTensorDataForView(const Tensor& src, MPSShape *mpsShape, const MPSDataType mpsDataType) {
  IntArrayRef src_base_shape = get_buffer_shape(src.storage().data());
  std::vector<int64_t> src_view_shape;
  bool hasMPSShape = (mpsShape != nil);
  int src_ndim_base = src_base_shape.size();
  int src_ndim_view = 0;
  if (hasMPSShape) {
    src_ndim_view = [mpsShape count];
    src_view_shape.reserve(src_ndim_view);
    for (const auto i : c10::irange(src_ndim_view)) {
      src_view_shape[i] = [mpsShape[i] intValue];
    }
  } else {
    src_ndim_view = src.dim();
    src_view_shape = src.sizes().vec();
  }

  MPSNDArray *srcTensorNDArrayView = nil;
  MPSNDArrayDescriptor *srcTensorNDArrayDesc = nil;
  MPSNDArray *srcTensorNDArray = nil;
  id<MTLCommandBuffer> commandBuffer = getCurrentMPSStream()->commandBuffer();

  if (src_ndim_base == src_ndim_view) {
    srcTensorNDArray = ndArrayFromTensor(src, getMPSShape(src_base_shape), mpsDataType);
    srcTensorNDArrayDesc = srcTensorNDArray.descriptor;

    int firstDimToSlice = 0;
    while (src_base_shape[firstDimToSlice] == src_view_shape[firstDimToSlice]) {
      firstDimToSlice++;
    }

    int view_numel = 1;
    for (const auto i : c10::irange(firstDimToSlice + 1, src_base_shape.size())) {
      view_numel *= src_base_shape[i];
    }

    int sliceOffset = src.storage_offset() / view_numel;
    // There are cases where both dimensions of a view can shrink
    // E.g: x = torch.randn((3,6))[1, 1:3]
    int nextSliceOffset = src.storage_offset() % view_numel;

    [srcTensorNDArrayDesc sliceDimension:src_ndim_base - 1 - firstDimToSlice withSubrange:{static_cast<NSUInteger>(sliceOffset), static_cast<NSUInteger>(src.sizes()[firstDimToSlice])}];
    if (nextSliceOffset) {
      [srcTensorNDArrayDesc sliceDimension:src_ndim_base - 2 - firstDimToSlice withSubrange:{static_cast<NSUInteger>(nextSliceOffset), static_cast<NSUInteger>(src.sizes()[firstDimToSlice+1])}];
    }
  }
  else {
    int src_view_numel = 1;
    for (const auto i : c10::irange(src_ndim_view)) {
      src_view_numel *= src_view_shape[i];
    }

    int idx = 0;
    int finalShapeSize = (src_ndim_view == 0) ? 1 : src_ndim_view;
    std::vector<NSNumber*> mpsFinalShape(finalShapeSize);

    // When the shapes are different, we need to flatten the first slice in order to alias the memory without any copies
    // E.g: base tensor [5, 7, 3], view tensor [7, 3] (storage_offset=21). We need to flatten [5, 7, 3] to [35, 3], then
    // we can slice directly into the first dimension based on the storage_offset
    uint32_t flattenedSlice = 1;
    for (const auto i : c10::irange(src_ndim_base - finalShapeSize + 1)) {
      flattenedSlice *= src_base_shape[i];
    }
    mpsFinalShape[idx++] = [NSNumber numberWithInteger:flattenedSlice];

    for (const auto i : c10::irange(src_ndim_base - finalShapeSize + 1, src_ndim_base)) {
      mpsFinalShape[idx++] = [NSNumber numberWithInteger:src_base_shape[i]];
    }

    mpsShape = [NSArray arrayWithObjects:mpsFinalShape.data() count:mpsFinalShape.size()];
    srcTensorNDArray = ndArrayFromTensor(src, mpsShape, mpsDataType);
    srcTensorNDArrayDesc = srcTensorNDArray.descriptor;

    int dim0 = (src_ndim_view == 0) ? 1 : src_view_shape[0];
    int totalSlices = dim0;

    // For 1D arrays, the storage_offset gives directly the
    // starting point from where the slice should start
    int sliceOffset = src_ndim_view == 1 ? 1 : dim0;
    int view_numel = src_ndim_view == 1 ? 1 : src_view_numel;
    [srcTensorNDArrayDesc sliceDimension:finalShapeSize - 1 withSubrange:{static_cast<NSUInteger>((src.storage_offset() / view_numel) * sliceOffset), static_cast<NSUInteger>(totalSlices)}];
  }

  srcTensorNDArrayView = [srcTensorNDArray arrayViewWithCommandBuffer:commandBuffer
                                                           descriptor:srcTensorNDArrayDesc
                                                             aliasing:MPSAliasingStrategyShallAlias];

  return [[[MPSGraphTensorData alloc] initWithMPSNDArray:srcTensorNDArrayView] autorelease];
}

MPSGraphTensor *permuteTensor(MPSGraph *graph, MPSGraphTensor *inputTensor, NSArray *permuteOrder) {
  NSUInteger srcRank = [[inputTensor shape] count];
  if (srcRank != [permuteOrder count]) {
    return nil;
  }

  MPSGraphTensor *outputTensor = inputTensor;
  std::vector<NSUInteger> dimensionOrder(srcRank);
  std::iota (std::begin(dimensionOrder), std::end(dimensionOrder), 0);

  for (const auto  i : c10::irange(srcRank)) {
    NSUInteger axis = [permuteOrder[i] integerValue];
    auto axisIter = std::find(dimensionOrder.begin(), dimensionOrder.end(), axis);
    NSUInteger axis1 = i;
    NSUInteger axis2 = axisIter - dimensionOrder.begin();
    iter_swap(dimensionOrder.begin() + i, axisIter);

    outputTensor = [graph transposeTensor:outputTensor
                                dimension:axis1
                            withDimension:axis2
                                     name:nil];
  }

  return outputTensor;
}

NSDictionary *getStrideToDimLengthOffsetDict(MPSGraphTensor *tensor, NSUInteger rank, NSUInteger offset) {
  // Assuming input tensor has default strides
  NSInteger stride = 1;
  NSMutableDictionary *strideToDimLengthOffset = [[NSMutableDictionary alloc] init];
  for (NSInteger srcDim = rank - 1; srcDim >= 0; srcDim--) {
    NSUInteger size = [[tensor shape][srcDim] integerValue];
    NSDictionary *entry =
    @{
      @"dim": [NSNumber numberWithInteger:srcDim],
      @"length": [tensor shape][srcDim],
      @"offset": [NSNumber numberWithInteger:offset % size] // offset is determined traversing backwards through stride
    };
    [strideToDimLengthOffset setValue:entry forKey:[NSString stringWithFormat:@"%ld",stride]];
    offset /= size;
    stride *= size;
  }
  return strideToDimLengthOffset;
}

// Detect only expand dims, allows for duplicate strides
MPSGraphTensor* asStridedLayer_expandDimsPattern(MPSGraph *graph, MPSGraphTensor *inputTensor, int dstRank, const IntArrayRef& dstSizes, const IntArrayRef& dstStrides, int offset) {

  NSUInteger srcRank = [[inputTensor shape] count];
  // Not an expand dims
  if (srcRank >= dstRank)
    return nil;

  NSMutableArray *expandAxes = [[NSMutableArray alloc] init];

  BOOL isValidExpand = YES;
  NSInteger currSrcDim = (NSInteger)srcRank - 1;
  NSUInteger currSrcStride = 1;
  for (NSInteger dstDim = dstRank - 1; dstDim >= 0 && isValidExpand; dstDim--) {
    NSUInteger currDimLength = dstSizes[dstDim];
    NSUInteger currStride = dstStrides[dstDim];
    NSUInteger currSrcDimLength = currSrcDim >= 0 ? [[inputTensor shape][currSrcDim] integerValue] : 1;

    NSUInteger targetDimLength =  currSrcDimLength;
    if (currDimLength != targetDimLength) {
      targetDimLength = 1;
    }
    if (currDimLength != targetDimLength || currStride != currSrcStride) {
      isValidExpand = NO;
    }
    if (currSrcDim >= 0 && currSrcDimLength == targetDimLength) {
      currSrcStride *= currSrcDimLength;
      currSrcDim--;
    } else {
      [expandAxes addObject:[NSNumber numberWithInt:dstDim]];
    }
  }

  // Did not use every dimension of source
  if (!isValidExpand || currSrcDim >= 0) {
    [expandAxes release];
    return nil;
  }

  MPSGraphTensor *expandTensor = inputTensor;
  if ([expandAxes count]) {
    expandTensor = [graph expandDimsOfTensor:expandTensor
                                        axes:expandAxes
                                        name:nil];
  }
  [expandAxes release];

  return expandTensor;
}

// Detect contiguous reshapes, no slicing
MPSGraphTensor* asStridedLayer_reshapePattern(MPSGraph *graph, MPSGraphTensor *inputTensor, int dstRank, const IntArrayRef& dstSizes, const IntArrayRef& dstStrides, int offset) {
  NSUInteger srcRank = [[inputTensor shape] count];
  // Not a reshape
  if (srcRank <= dstRank)
    return nil;

  NSMutableArray *dstShape = [[NSMutableArray alloc] init];

  BOOL isValidReshape = YES;
  NSInteger srcDim = srcRank - 1;
  NSUInteger srcStride = 1;
  for (NSInteger dstDim = dstRank - 1; dstDim >= 0 && isValidReshape; dstDim--) {
    NSUInteger currDimLength = dstSizes[dstDim];
    NSUInteger currStride = dstStrides[dstDim];
    [dstShape insertObject:[NSNumber numberWithInteger:currDimLength] atIndex: 0];

    NSUInteger targetDimLength = currDimLength;
    NSUInteger currReshapeSize = 1;
    NSUInteger innerStride = srcStride;

    while (currReshapeSize != targetDimLength && srcDim >= 0) {
      NSUInteger srcDimLength = [[inputTensor shape][srcDim] integerValue];
      currReshapeSize *= srcDimLength;
      srcStride *= srcDimLength;
      srcDim--;
    };

    isValidReshape &= (currReshapeSize == targetDimLength && currStride == innerStride);
  }
  isValidReshape &= (srcDim < 0);

  MPSGraphTensor *outputTensor = nil;
  if (isValidReshape)
    outputTensor = [graph reshapeTensor: inputTensor
                              withShape: dstShape
                                   name: nil];
  [dstShape release];
  return outputTensor;
}

MPSGraphTensor* asStridedLayer_genericPattern(MPSGraph *graph, MPSGraphTensor *inputTensor, int dstRank, const IntArrayRef& dstSizes, const IntArrayRef& dstStrides, int offset) {

  // Duplicate strides cannot be done
  {
    BOOL allUnique = YES;
    NSMutableSet *uniqueStrides = [[NSMutableSet alloc] init];
    for (NSInteger dstDim = 0; (dstDim < dstRank) && allUnique; dstDim++) {
      int stride = dstStrides[dstDim];
      NSNumber *strideObj = [NSNumber numberWithInt:stride];
      allUnique &= (stride == 0 || ![uniqueStrides containsObject:strideObj]);
      [uniqueStrides addObject: strideObj];
    }
    [uniqueStrides release];
    if (!allUnique)
      return nil;

    // Skip for zero in dst shape
    for (NSInteger dstDim = 0; dstDim < dstRank; dstDim++)
      if (dstSizes[dstDim] == 0) { return nil; }
  }

  // 1. Flatten the inputTensor if neccessary
  MPSGraphTensor *flatInputTensor = inputTensor;
  {
    // Flatten inputs to remove duplicate strides.
    NSMutableArray *squeezeAxes = [[NSMutableArray alloc] init];
    for(NSUInteger srcDim = 1; srcDim < [[flatInputTensor shape] count]; srcDim++) {
        if ([[flatInputTensor shape][srcDim] intValue] == 1)
            [squeezeAxes addObject:[NSNumber numberWithInteger:srcDim]];
    }
    // We have to leave at least 1 dimension, if all input dims are 1
    if ([squeezeAxes count])
        flatInputTensor = [graph squeezeTensor:flatInputTensor
                                          axes:squeezeAxes
                                          name:nil];
    [squeezeAxes release];
  }

  int srcRank = (int)[[flatInputTensor shape] count];
  NSDictionary *srcStrideToDimLengthOffset = getStrideToDimLengthOffsetDict(flatInputTensor, srcRank, offset);

  // Populate the dimension order, slice info, and broadcast info
  NSMutableArray *dstDimOrder = [[NSMutableArray alloc] init];
  std::vector<int32_t> dstDimToSliceLength(dstRank);
  std::vector<int32_t> dstDimToSliceOffset(dstRank);
  bool needsBroadcast = false;
  {
    for (NSInteger dstDim = dstRank - 1; dstDim >= 0; dstDim--) {
      if (dstStrides[dstDim] == 0) {
        // This dimension should be a broadcast
        needsBroadcast = true;
        dstDimToSliceLength[dstDim] = dstSizes[dstDim];
        dstDimToSliceOffset[dstDim] = 0;
      } else {
        // Find what dimension and native length was for the specified stride
        NSDictionary *srcDimLengthOffset = srcStrideToDimLengthOffset[[NSString stringWithFormat:@"%lld",dstStrides[dstDim]]];

        // Stride does not exist in source tensor, or the specified size is too long. Not possible
        // TODO: Longer length with same stride + removal of dim(s) above this is a flatten/reshape. Consider adding support
        if (!srcDimLengthOffset || dstSizes[dstDim] > [srcDimLengthOffset[@"length"] intValue])
          return nil;

        // Get the src dimension corresponding to the requested stride
        NSNumber *srcDim = srcDimLengthOffset[@"dim"];
        [dstDimOrder insertObject:srcDim atIndex:0];

        dstDimToSliceLength[dstDim] = dstSizes[dstDim];
        dstDimToSliceOffset[dstDim] = [srcDimLengthOffset[@"offset"] intValue];
      }
    }
  }

  // 2. Slice out any unused dimensions
  NSMutableArray *missingSrcDims = [[NSMutableArray alloc] init];
  MPSGraphTensor *slicedUnusedTensor = flatInputTensor;
  {
    // Find any src strides/dims that are not present in the dst
    NSMutableArray *missingSrcStrides = [[NSMutableArray alloc] init];
    {
      NSUInteger stride = 1;
      for (NSInteger srcDim = [[flatInputTensor shape] count] - 1; srcDim >= 0; srcDim--) {
        [missingSrcStrides addObject:[NSNumber numberWithInteger:stride]];
        stride *= [[flatInputTensor shape][srcDim] integerValue];
      }
      for (NSInteger dstDim = 0; dstDim < dstRank; dstDim++) {
        [missingSrcStrides removeObject:[NSNumber numberWithInteger:dstStrides[dstDim]]];
      }
    }
    for (NSUInteger i = 0; i < [missingSrcStrides count]; i++) {
      NSUInteger stride = [missingSrcStrides[i] integerValue];
      NSDictionary *srcDimLengthOffset = srcStrideToDimLengthOffset[[NSString stringWithFormat:@"%ld",stride]];
      NSNumber *missingSrcDim = srcDimLengthOffset[@"dim"];
      [missingSrcDims addObject:missingSrcDim];
      [dstDimOrder insertObject:missingSrcDim atIndex:0];

      slicedUnusedTensor = [graph sliceTensor:slicedUnusedTensor
                                    dimension:[missingSrcDim intValue]
                                        start:[srcDimLengthOffset[@"offset"] intValue]
                                       length:1
                                         name:nil];
    }
    [missingSrcStrides release];
  }

  // 3. Transpose if necessary
  MPSGraphTensor *transposedTensor = slicedUnusedTensor;
  {
    // TODO: Use Transpose API
    BOOL needsTranspose = NO;
    for(NSUInteger dstDim = 0; dstDim < [dstDimOrder count] && !needsTranspose; dstDim++ )
      needsTranspose |= ([dstDimOrder[dstDim] intValue] != dstDim);
    if (needsTranspose)
      transposedTensor = permuteTensor(graph, transposedTensor, dstDimOrder);
  }

  // 4. Squeeze any unused dimensions following transpose
  MPSGraphTensor *squeezedTensor = transposedTensor;
  {
    // Transpose the missing dims back
    NSMutableArray *transposedMissingSrcDims = [[NSMutableArray alloc] init];
    for (NSUInteger dstDim = 0; dstDim < [dstDimOrder count]; dstDim++) {
      NSNumber *srcDim = dstDimOrder[dstDim];
      if ([missingSrcDims containsObject:srcDim])
        [transposedMissingSrcDims addObject:[NSNumber numberWithInt:dstDim]];
    }
    if ([transposedMissingSrcDims count])
      squeezedTensor = [graph squeezeTensor:squeezedTensor
                                       axes:transposedMissingSrcDims
                                       name:nil];
    [transposedMissingSrcDims release];
  }

  // 5. Slice
  MPSGraphTensor *slicedTensor = squeezedTensor;
  {
    NSUInteger currDstDim = 0;
    for (NSUInteger dstDim = 0; dstDim < dstRank; dstDim++) {
      // Only dstDims with nonzero stride are in the current tensor, skip broadcasts
      if (dstStrides[dstDim] != 0) {
        int start = dstDimToSliceOffset[dstDim];
        int length = dstDimToSliceLength[dstDim];
        if (length != [[slicedTensor shape][currDstDim] intValue])
          slicedTensor = [graph sliceTensor:slicedTensor
                                  dimension:currDstDim
                                      start:start
                                     length:length
                                       name:nil];
        currDstDim++;
      }
    }
  }

  // 6. Expand then broadcast the source tensor
  MPSGraphTensor *broadcastTensor = slicedTensor;
  if (needsBroadcast) {
    NSMutableArray *broadcastShape = [[NSMutableArray alloc] init];
    NSMutableArray *expandAxes = [[NSMutableArray alloc] init];
    for(NSInteger dstDim = 0; dstDim < dstRank; dstDim++) {
      [broadcastShape addObject:[NSNumber numberWithInt:dstSizes[dstDim]]];
      if (dstStrides[dstDim] == 0)
        [expandAxes addObject:[NSNumber numberWithInt:dstDim]];
    }

    if ([expandAxes count]) {
      MPSGraphTensor *expandTensor = [graph expandDimsOfTensor:broadcastTensor
                                                          axes:expandAxes
                                                          name:nil];
      broadcastTensor = [graph broadcastTensor:expandTensor
                                       toShape:broadcastShape
                                          name:nil];
    }
    [broadcastShape release];
    [expandAxes release];
  }

  [srcStrideToDimLengthOffset release];
  [dstDimOrder release];
  [missingSrcDims release];

  return broadcastTensor;
}

MPSGraphTensor* asStridedLayer_pattern(MPSGraph *graph, MPSGraphTensor *inputTensor, int dstRank, const IntArrayRef& dstSizes, const IntArrayRef& dstStrides, int offset) {
  if (!dstRank)
    return nil;

  MPSGraphTensor *outputTensor = nil;
  outputTensor = asStridedLayer_expandDimsPattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);
  if (!outputTensor)
    outputTensor = asStridedLayer_reshapePattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);
  if (!outputTensor)
    outputTensor = asStridedLayer_genericPattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);

  return outputTensor;
}


static MPSGraphTensor* chainViewOperation(ViewCachedGraph* cachedGraph, const IntArrayRef& size,
                                          const IntArrayRef& stride, int64_t offset,
                                          const IntArrayRef& base_shape, bool needsScatter,
                                          const bool needsBoolCast,
                                          MPSGraphTensor* updatesTensor)
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

    if (!needsScatter) {
      MPSGraphTensor *outputTensor = asStridedLayer_pattern(mpsGraph, inputTensor, shape_size, size, stride, offset);

      if (outputTensor) {
        if (needsBoolCast) {
          outputTensor = [mpsGraph castTensor:outputTensor
                                       toType:MPSDataTypeBool
                                         name:@"Cast back to bool"];
        }
        return outputTensor;
      }
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
                                                     updatesTensor: updatesTensor
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

static IntArrayRef updateTensorBaseShape(const Tensor& self)
{
  IntArrayRef base_shape = get_buffer_shape(self.storage().data());
  // if there's no base_shape stored in MPSAllocator, then infer it from tensor's size and store it
  if (base_shape.size() == 0) {
    // IntArrayRef wouldn't own the data, so we use a static storage
    static const int64_t shape_1d = 1;
    // self.sizes().size() could be zero
    base_shape = self.sizes().size() ? self.sizes() :
                ((self.is_view() && self._base().sizes().size()) ? self._base().sizes() : IntArrayRef(&shape_1d, 1));

    // base_shape will be retained in MPSAllocator until buffer gets recycled
    if (self.storage().data())
      set_buffer_shape(self.storage().data(), base_shape);
  }
  return base_shape;
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
static ViewCachedGraph* createViewGraph(const Tensor& self, const Tensor &updates, IntArrayRef size, IntArrayRef stride, int64_t storage_offset, bool needsScatter)
{
  IntArrayRef base_shape = updateTensorBaseShape(self);

  @autoreleasepool {
    string key = getStridedKey(self.scalar_type(), updates.scalar_type(), base_shape, size, stride, storage_offset, needsScatter);
    MPSGraphCache* cache_ = MPSGraphCache::getInstance();
    ViewCachedGraph* cachedGraph = static_cast<ViewCachedGraph *>(cache_->LookUp(key));

    if (!cachedGraph) {
      cachedGraph = static_cast<ViewCachedGraph *>(cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        ViewCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            MPSGraphTensor* updatesTensor = nil;
            newCachedGraph = new ViewCachedGraph(mpsGraph);
            // Workaround for MPSShaderLibrary bug
            // TODO: Remove once https://github.com/pytorch/pytorch/issues/82305 is resolved
            auto inputType = getMPSScalarType(self.scalar_type());
            if (inputType == MPSDataTypeUInt8) {
                inputType = MPSDataTypeInt8;
            }
            auto needsBoolCast = inputType == MPSDataTypeBool;
            // Self is the input tensor we are creating view of
            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(base_shape));
            newCachedGraph->storageOffsetTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]);
            for (int i = 0; i < size.size(); i++) {
              newCachedGraph->strideTensors.push_back(mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[@1]));
            }
            if (needsScatter) {
              auto updatesType = getMPSScalarType(updates.scalar_type());
              if (updatesType == MPSDataTypeUInt8) {
                updatesType = MPSDataTypeInt8;
              }
              newCachedGraph->updatesTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, updatesType);
              updatesTensor = newCachedGraph->updatesTensor;
              if (inputType != updatesType) {
                updatesTensor = [mpsGraph castTensor:updatesTensor
                                              toType:inputType
                                                name:@"castUpdatesTensor"];
              }
            }
            newCachedGraph->outputTensor = chainViewOperation(newCachedGraph, size, stride, storage_offset, base_shape, needsScatter, needsBoolCast, updatesTensor);
        }
        return newCachedGraph;
      }));
    }
    return cachedGraph;
  }
}

Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst)
{
  if (src.sizes().size() == 0) {
    return Tensor();
  }
  bool requires_sync = false;
  Tensor output;
  if (!dst.has_storage()) {
    output = at::native::empty_mps(src.sizes(), src.scalar_type(), c10::nullopt, kMPS);
    requires_sync = true;
  }
  ViewCachedGraph* cachedGraph = createViewGraph(src, dst, src.sizes(), src.strides(),
                                                 src.storage_offset(), /*needsScatter*/ false);
  return runViewGraph(cachedGraph, src, dst.has_storage() ? dst : output, /*needsScatter*/ false, requires_sync);
}

Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  ViewCachedGraph* cachedGraph = createViewGraph(output, src, output.sizes(), output.strides(),
                                                 output.storage_offset(), /*needsScatter*/ true);
  return runViewGraph(cachedGraph, src, output, /*needsScatter*/ true, /*requires_sync*/  true);
}

} // namespace mps

// implementation of as_strided() op
Tensor as_strided_tensorimpl_mps(const Tensor& self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // creating the view graph will be deferred until gatherViewTensor() or scatterViewTensor() are called.
  // In as_strided, we just update the base shape of the buffer in order to retrieve it later
  // when we create/run the view graph.
  IntArrayRef base_shape = mps::updateTensorBaseShape(self);
  TORCH_INTERNAL_ASSERT(base_shape.size() > 0, "Failed to update the base shape of tensor's buffer at ", self.storage().data());

  return result;
}

} // namespace at::native
