//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/IndexKernels.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
namespace mps {

struct ViewCachedGraph : public MPSCachedGraph {
  ViewCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor = nil;
  MPSGraphTensor* outputTensor = nil;
  MPSGraphTensor* updatesTensor = nil;
  MPSGraphTensor* storageOffsetTensor = nil;
  std::vector<MPSGraphTensor*> strideTensors;
};

static std::string getStridedKey(const ScalarType& self_dtype,
                                 const ScalarType& updates_dtype,
                                 const IntArrayRef& base_shape,
                                 const IntArrayRef& new_shape,
                                 const IntArrayRef& stride,
                                 int64_t storage_offset,
                                 bool is_scatter) {
  std::string dtype_key = getMPSTypeString(self_dtype);
  if (is_scatter) {
    dtype_key += ":" + getMPSTypeString(updates_dtype);
  }

  return (is_scatter ? "scatter:" : "gather:") + dtype_key + "[" + getArrayRefString(base_shape) + "]:[" +
      getArrayRefString(new_shape) + "]:[" + getArrayRefString(stride) + "]:[" + to_string(storage_offset) + "]";
}

// initializes the MTLBuffers for tensor data and runs the MPSGraph for the view op
static Tensor& runViewGraph(ViewCachedGraph* cachedGraph, const at::Tensor& src, Tensor& output, bool needsScatter) {
  const id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  const id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);

  const IntArrayRef& strides = needsScatter ? output.strides() : src.strides();
  const IntArrayRef& sizes = needsScatter ? output.sizes() : src.sizes();
  const int64_t storage_offset = needsScatter ? output.storage_offset() : src.storage_offset();
  const MPSDataType inputType = [cachedGraph->inputTensor dataType];

  MPSShape* inputShape = [cachedGraph->inputTensor shape];
  MPSShape* outputShape = needsScatter ? inputShape : getMPSShape(src);

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    // in case of scatter, we use output tensor as input buffer and write the results back to the source buffer
    feeds[cachedGraph->inputTensor] =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:needsScatter ? outputBuffer : sourceBuffer
                                                 shape:inputShape
                                              dataType:inputType] autorelease];
    if (needsScatter) {
      auto updatesType = getMPSScalarType(src.scalar_type());
      if (updatesType == MPSDataTypeUInt8 || (updatesType == MPSDataTypeBool && !is_macos_13_or_newer())) {
        updatesType = MPSDataTypeInt8;
      }

      feeds[cachedGraph->updatesTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer:sourceBuffer
                                                                                   shape:getMPSShape(src.numel())
                                                                                dataType:updatesType] autorelease];
    }
    MPSScalar storageOffsetScalar = getMPSScalar(storage_offset, ScalarType::Int);
    feeds[cachedGraph->storageOffsetTensor] = getMPSGraphTensorFromScalar(stream, storageOffsetScalar);

    std::vector<MPSScalar> strideScalars(sizes.size());
    for (const auto i : c10::irange(sizes.size())) {
      strideScalars[i] = getMPSScalar(strides[i], ScalarType::Int);
      feeds[cachedGraph->strideTensors[i]] = getMPSGraphTensorFromScalar(stream, strideScalars[i]);
    }
    // Workaround for MPSShaderLibrary bug in macOS Monterey
    // This is fixed in macOS Ventura
    auto outputType = getMPSScalarType(output.scalar_type());
    if (outputType == MPSDataTypeUInt8 || (outputType == MPSDataTypeBool && !is_macos_13_or_newer())) {
      outputType = MPSDataTypeInt8;
    }
    MPSGraphTensorData* outputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                                    shape:outputShape
                                                                                 dataType:outputType] autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{cachedGraph->outputTensor : outputTensorData};
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
  return output;
}

MPSGraphTensor* permuteTensor(MPSGraph* graph, MPSGraphTensor* inputTensor, NSArray* permuteOrder) {
  NSUInteger srcRank = [[inputTensor shape] count];
  if (srcRank != [permuteOrder count]) {
    return nil;
  }

  MPSGraphTensor* outputTensor = inputTensor;
  std::vector<NSUInteger> dimensionOrder(srcRank);
  std::iota(std::begin(dimensionOrder), std::end(dimensionOrder), 0);

  for (const auto i : c10::irange(srcRank)) {
    NSUInteger axis = [permuteOrder[i] integerValue];
    auto axisIter = std::find(dimensionOrder.begin(), dimensionOrder.end(), axis);
    NSUInteger axis1 = i;
    NSUInteger axis2 = axisIter - dimensionOrder.begin();
    iter_swap(dimensionOrder.begin() + i, axisIter);

    outputTensor = [graph transposeTensor:outputTensor dimension:axis1 withDimension:axis2 name:nil];
  }

  return outputTensor;
}

NSDictionary* getStrideToDimLengthOffsetDict(MPSGraphTensor* tensor, NSUInteger rank, NSUInteger offset) {
  // Assuming input tensor has default strides
  NSInteger stride = 1;
  NSMutableDictionary* strideToDimLengthOffset = [[NSMutableDictionary alloc] init];
  for (NSInteger srcDim = rank - 1; srcDim >= 0; srcDim--) {
    NSUInteger size = [[tensor shape][srcDim] integerValue];
    NSDictionary* entry = @{
      @"dim" : [NSNumber numberWithInteger:srcDim],
      @"length" : [tensor shape][srcDim],
      @"offset" : [NSNumber numberWithInteger:offset % size] // offset is determined traversing backwards through stride
    };
    [strideToDimLengthOffset setValue:entry forKey:[NSString stringWithFormat:@"%ld", stride]];
    offset /= size;
    stride *= size;
  }
  return strideToDimLengthOffset;
}

// Detect only expand dims, allows for duplicate strides
MPSGraphTensor* asStridedLayer_expandDimsPattern(MPSGraph* graph,
                                                 MPSGraphTensor* inputTensor,
                                                 size_t dstRank,
                                                 const IntArrayRef& dstSizes,
                                                 const IntArrayRef& dstStrides,
                                                 int offset) {
  NSUInteger srcRank = [[inputTensor shape] count];
  // Not an expand dims
  if (srcRank >= dstRank)
    return nil;

  NSMutableArray* expandAxes = [[NSMutableArray alloc] init];

  BOOL isValidExpand = YES;
  NSInteger currSrcDim = (NSInteger)srcRank - 1;
  NSUInteger currSrcStride = 1;
  for (NSInteger dstDim = dstRank - 1; dstDim >= 0 && isValidExpand; dstDim--) {
    NSUInteger currDimLength = dstSizes[dstDim];
    NSUInteger currStride = dstStrides[dstDim];
    NSUInteger currSrcDimLength = currSrcDim >= 0 ? [[inputTensor shape][currSrcDim] integerValue] : 1;

    NSUInteger targetDimLength = currSrcDimLength;
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

  MPSGraphTensor* expandTensor = inputTensor;
  if ([expandAxes count]) {
    expandTensor = [graph expandDimsOfTensor:expandTensor axes:expandAxes name:nil];
  }
  [expandAxes release];

  return expandTensor;
}

// Detect contiguous reshapes, no slicing
MPSGraphTensor* asStridedLayer_reshapePattern(MPSGraph* graph,
                                              MPSGraphTensor* inputTensor,
                                              size_t dstRank,
                                              const IntArrayRef& dstSizes,
                                              const IntArrayRef& dstStrides,
                                              int offset) {
  NSUInteger srcRank = [[inputTensor shape] count];
  // Not a reshape
  if (srcRank <= dstRank)
    return nil;

  NSMutableArray* dstShape = [[NSMutableArray alloc] init];

  BOOL isValidReshape = YES;
  NSInteger srcDim = srcRank - 1;
  NSUInteger srcStride = 1;
  for (NSInteger dstDim = dstRank - 1; dstDim >= 0 && isValidReshape; dstDim--) {
    NSUInteger currDimLength = dstSizes[dstDim];
    NSUInteger currStride = dstStrides[dstDim];
    [dstShape insertObject:[NSNumber numberWithInteger:currDimLength] atIndex:0];

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

  MPSGraphTensor* outputTensor = nil;
  if (isValidReshape)
    outputTensor = [graph reshapeTensor:inputTensor withShape:dstShape name:nil];
  [dstShape release];
  return outputTensor;
}

MPSGraphTensor* asStridedLayer_genericPattern(MPSGraph* graph,
                                              MPSGraphTensor* inputTensor,
                                              size_t dstRank,
                                              const IntArrayRef& dstSizes,
                                              const IntArrayRef& dstStrides,
                                              int offset) {
  // Duplicate strides cannot be done
  {
    BOOL allUnique = YES;
    NSMutableSet* uniqueStrides = [[NSMutableSet alloc] init];
    for (NSUInteger dstDim = 0; (dstDim < dstRank) && allUnique; dstDim++) {
      int stride = dstStrides[dstDim];
      NSNumber* strideObj = [NSNumber numberWithInt:stride];
      allUnique &= (stride == 0 || ![uniqueStrides containsObject:strideObj]);
      [uniqueStrides addObject:strideObj];
    }
    [uniqueStrides release];
    if (!allUnique)
      return nil;

    // Skip for zero in dst shape
    for (NSUInteger dstDim = 0; dstDim < dstRank; dstDim++)
      if (dstSizes[dstDim] == 0) {
        return nil;
      }
  }

  // 1. Flatten the inputTensor if necessary
  MPSGraphTensor* flatInputTensor = inputTensor;
  {
    // Flatten inputs to remove duplicate strides.
    NSMutableArray* squeezeAxes = [[NSMutableArray alloc] init];
    for (NSUInteger srcDim = 1; srcDim < [[flatInputTensor shape] count]; srcDim++) {
      if ([[flatInputTensor shape][srcDim] intValue] == 1)
        [squeezeAxes addObject:[NSNumber numberWithInteger:srcDim]];
    }
    // We have to leave at least 1 dimension, if all input dims are 1
    if ([squeezeAxes count])
      flatInputTensor = [graph squeezeTensor:flatInputTensor axes:squeezeAxes name:nil];
    [squeezeAxes release];
  }

  int srcRank = (int)[[flatInputTensor shape] count];
  NSDictionary* srcStrideToDimLengthOffset = getStrideToDimLengthOffsetDict(flatInputTensor, srcRank, offset);

  // Populate the dimension order, slice info, and broadcast info
  NSMutableArray* dstDimOrder = [[NSMutableArray alloc] init];
  std::vector<int32_t> dstDimToSliceLength(dstRank);
  std::vector<int32_t> dstDimToSliceOffset(dstRank);
  bool needsBroadcast = false;
  {
    for (auto dstDim = dstRank - 1; dstDim >= 0; dstDim--) {
      if (dstStrides[dstDim] == 0) {
        // This dimension should be a broadcast
        needsBroadcast = true;
        dstDimToSliceLength[dstDim] = dstSizes[dstDim];
        dstDimToSliceOffset[dstDim] = 0;
      } else {
        // Find what dimension and native length was for the specified stride
        NSDictionary* srcDimLengthOffset =
            srcStrideToDimLengthOffset[[NSString stringWithFormat:@"%lld", dstStrides[dstDim]]];

        dstDimToSliceLength[dstDim] = dstSizes[dstDim];
        dstDimToSliceOffset[dstDim] = [srcDimLengthOffset[@"offset"] intValue];

        // Stride does not exist in source tensor, or the specified size is too long. Not possible
        // TODO: Longer length with same stride + removal of dim(s) above this is a flatten/reshape. Consider adding
        // support
        if (!srcDimLengthOffset ||
            // the offset + length of destination should not be larger than source's length when slicing
            dstDimToSliceOffset[dstDim] + dstDimToSliceLength[dstDim] > [srcDimLengthOffset[@"length"] intValue]) {
          return nil;
        }
        // Get the src dimension corresponding to the requested stride
        NSNumber* srcDim = srcDimLengthOffset[@"dim"];
        [dstDimOrder insertObject:srcDim atIndex:0];
      }
    }
  }

  // 2. Slice out any unused dimensions
  NSMutableArray* missingSrcDims = [[NSMutableArray alloc] init];
  MPSGraphTensor* slicedUnusedTensor = flatInputTensor;
  {
    // Find any src strides/dims that are not present in the dst
    NSMutableArray* missingSrcStrides = [[NSMutableArray alloc] init];
    {
      NSUInteger stride = 1;
      for (NSInteger srcDim = [[flatInputTensor shape] count] - 1; srcDim >= 0; srcDim--) {
        [missingSrcStrides addObject:[NSNumber numberWithInteger:stride]];
        stride *= [[flatInputTensor shape][srcDim] integerValue];
      }
      for (NSUInteger dstDim = 0; dstDim < dstRank; dstDim++) {
        [missingSrcStrides removeObject:[NSNumber numberWithInteger:dstStrides[dstDim]]];
      }
    }
    for (NSUInteger i = 0; i < [missingSrcStrides count]; i++) {
      NSUInteger stride = [missingSrcStrides[i] integerValue];
      NSDictionary* srcDimLengthOffset = srcStrideToDimLengthOffset[[NSString stringWithFormat:@"%ld", stride]];
      NSNumber* missingSrcDim = srcDimLengthOffset[@"dim"];
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
  MPSGraphTensor* transposedTensor = slicedUnusedTensor;
  {
    // TODO: Use Transpose API
    BOOL needsTranspose = NO;
    for (NSUInteger dstDim = 0; dstDim < [dstDimOrder count] && !needsTranspose; dstDim++)
      needsTranspose |= ([dstDimOrder[dstDim] intValue] != static_cast<int>(dstDim));
    if (needsTranspose)
      transposedTensor = permuteTensor(graph, transposedTensor, dstDimOrder);
  }

  // 4. Squeeze any unused dimensions following transpose
  MPSGraphTensor* squeezedTensor = transposedTensor;
  {
    // Transpose the missing dims back
    NSMutableArray* transposedMissingSrcDims = [[NSMutableArray alloc] init];
    for (NSUInteger dstDim = 0; dstDim < [dstDimOrder count]; dstDim++) {
      NSNumber* srcDim = dstDimOrder[dstDim];
      if ([missingSrcDims containsObject:srcDim])
        [transposedMissingSrcDims addObject:[NSNumber numberWithInt:dstDim]];
    }
    if ([transposedMissingSrcDims count])
      squeezedTensor = [graph squeezeTensor:squeezedTensor axes:transposedMissingSrcDims name:nil];
    [transposedMissingSrcDims release];
  }

  // 5. Slice
  MPSGraphTensor* slicedTensor = squeezedTensor;
  {
    NSUInteger currDstDim = 0;
    for (NSUInteger dstDim = 0; dstDim < dstRank; dstDim++) {
      // Only dstDims with nonzero stride are in the current tensor, skip broadcasts
      if (dstStrides[dstDim] != 0) {
        int start = dstDimToSliceOffset[dstDim];
        int length = dstDimToSliceLength[dstDim];
        if (length != [[slicedTensor shape][currDstDim] intValue])
          slicedTensor = [graph sliceTensor:slicedTensor dimension:currDstDim start:start length:length name:nil];
        currDstDim++;
      }
    }
  }

  // 6. Expand then broadcast the source tensor
  MPSGraphTensor* broadcastTensor = slicedTensor;
  if (needsBroadcast) {
    NSMutableArray* broadcastShape = [[NSMutableArray alloc] init];
    NSMutableArray* expandAxes = [[NSMutableArray alloc] init];
    for (NSUInteger dstDim = 0; dstDim < dstRank; dstDim++) {
      [broadcastShape addObject:[NSNumber numberWithInt:dstSizes[dstDim]]];
      if (dstStrides[dstDim] == 0)
        [expandAxes addObject:[NSNumber numberWithInt:dstDim]];
    }

    if ([expandAxes count]) {
      MPSGraphTensor* expandTensor = [graph expandDimsOfTensor:broadcastTensor axes:expandAxes name:nil];
      broadcastTensor = [graph broadcastTensor:expandTensor toShape:broadcastShape name:nil];
    }
    [broadcastShape release];
    [expandAxes release];
  }

  [srcStrideToDimLengthOffset release];
  [dstDimOrder release];
  [missingSrcDims release];

  return broadcastTensor;
}

MPSGraphTensor* asStridedLayer_pattern(MPSGraph* graph,
                                       MPSGraphTensor* inputTensor,
                                       size_t dstRank,
                                       const IntArrayRef& dstSizes,
                                       const IntArrayRef& dstStrides,
                                       int offset) {
  if (!dstRank)
    return nil;

  MPSGraphTensor* outputTensor = nil;
  outputTensor = asStridedLayer_expandDimsPattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);
  if (!outputTensor)
    outputTensor = asStridedLayer_reshapePattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);
  if (!outputTensor)
    outputTensor = asStridedLayer_genericPattern(graph, inputTensor, dstRank, dstSizes, dstStrides, offset);

  return outputTensor;
}

static std::vector<int64_t> getViewShape(const Tensor& src, MPSShape* mpsShape, const bool squeeze) {
  bool hasMPSShape = (mpsShape != nil);
  std::vector<int64_t> src_view_shape;
  if (hasMPSShape) {
    int src_ndim_view = [mpsShape count];
    if (squeeze) {
      for (const auto i : c10::irange(src_ndim_view)) {
        if ([mpsShape[i] intValue] == 1)
          continue;
        src_view_shape.emplace_back([mpsShape[i] intValue]);
      }
    } else {
      src_view_shape.resize(src_ndim_view);
      for (const auto i : c10::irange(src_ndim_view)) {
        src_view_shape[i] = [mpsShape[i] intValue];
      }
    }

  } else {
    if (squeeze) {
      IntArrayRef src_shape = src.sizes();
      size_t src_ndim_view = src_shape.size();
      for (const auto i : c10::irange(src_ndim_view)) {
        if (src_shape[i] == 1)
          continue;
        src_view_shape.emplace_back(src_shape[i]);
      }
    } else {
      src_view_shape = src.sizes().vec();
    }
  }

  return src_view_shape;
}

std::vector<int64_t> getSqueezedBaseShape(const Tensor& src, IntArrayRef shape) {
  std::vector<int64_t> src_base_shape;
  for (const auto i : c10::irange(shape.size())) {
    if (shape[i] == 1)
      continue;
    src_base_shape.emplace_back(shape[i]);
  }

  return src_base_shape;
}

bool canSliceViewTensor(const Tensor& src, MPSShape* mpsShape) {
  if (!src.is_contiguous()) {
    return false;
  }

  IntArrayRef src_base_shape = getIMPSAllocator()->getBufferShape(src.storage().data());
  size_t src_ndim_base = src_base_shape.size();
  std::vector<int64_t> src_view_shape = getViewShape(src, mpsShape, false);
  size_t src_ndim_view = src_view_shape.size();

  if (src_ndim_base != src_ndim_view) {
    return false;
  }

  for (const auto i : c10::irange(src_ndim_base)) {
    if (src_view_shape[i] > src_base_shape[i]) {
      return false;
    }
  }
  return true;
}

MPSGraphTensorData* getMPSGraphTensorDataForView(const Tensor& src, MPSShape* mpsShape, const MPSDataType mpsDataType) {
  IntArrayRef src_base_shape = getIMPSAllocator()->getBufferShape(src.storage().data());
  size_t src_ndim_base = src_base_shape.size();
  std::vector<int64_t> src_view_shape = getViewShape(src, mpsShape, false);
  size_t src_ndim_view = src_view_shape.size();

  MPSNDArray* srcTensorNDArrayView = nil;
  MPSNDArrayDescriptor* srcTensorNDArrayDesc = nil;
  MPSNDArray* srcTensorNDArray = nil;
  id<MTLCommandBuffer> commandBuffer = getCurrentMPSStream()->commandBuffer();
  size_t base_idx = 0;

  std::vector<int64_t> src_base_shape_vec;

  if (src_ndim_view != src_ndim_base) {
    src_base_shape_vec.reserve(src_ndim_view);
    for (const auto i : c10::irange(src_ndim_view)) {
      if (src_view_shape[i] == 1 && src_base_shape[base_idx] != 1) {
        src_base_shape_vec.emplace_back(1);
      } else {
        src_base_shape_vec.emplace_back(src_base_shape[base_idx]);
        if (base_idx < src_ndim_base - 1)
          base_idx += 1;
      }
    }
    src_base_shape = IntArrayRef(src_base_shape_vec);
    src_ndim_base = src_base_shape.size();
  }

  srcTensorNDArray = ndArrayFromTensor(src, getMPSShape(src_base_shape), mpsDataType);
  srcTensorNDArrayDesc = srcTensorNDArray.descriptor;

  size_t firstDimToSlice = 0;
  while (src_base_shape[firstDimToSlice] == src_view_shape[firstDimToSlice]) {
    firstDimToSlice++;
  }

  int64_t view_numel = 1;
  for (const auto i : c10::irange(firstDimToSlice + 1, src_base_shape.size())) {
    view_numel *= src_base_shape[i];
  }

  int64_t sliceOffset = src.storage_offset() / view_numel;
  [srcTensorNDArrayDesc
      sliceDimension:src_ndim_base - 1 - firstDimToSlice
        withSubrange:{static_cast<NSUInteger>(sliceOffset), static_cast<NSUInteger>(src.sizes()[firstDimToSlice])}];

  // Slice any remaining dimensions
  for (const auto crtSliceOffset : c10::irange(firstDimToSlice + 1, src_base_shape.size())) {
    if (src_view_shape[crtSliceOffset] != src_base_shape[crtSliceOffset]) {
      if (crtSliceOffset == src_base_shape.size() - 1) {
        sliceOffset = src.storage_offset() % src_base_shape[src_base_shape.size() - 1];
      } else {
        sliceOffset = (src.storage_offset() % view_numel) / (view_numel / src_base_shape[crtSliceOffset]);
      }
      [srcTensorNDArrayDesc
          sliceDimension:src_ndim_base - 1 - crtSliceOffset
            withSubrange:{static_cast<NSUInteger>(sliceOffset), static_cast<NSUInteger>(src.sizes()[crtSliceOffset])}];
    }
  }
  srcTensorNDArrayView = [srcTensorNDArray arrayViewWithCommandBuffer:commandBuffer
                                                           descriptor:srcTensorNDArrayDesc
                                                             aliasing:MPSAliasingStrategyShallAlias];

  return [[[MPSGraphTensorData alloc] initWithMPSNDArray:srcTensorNDArrayView] autorelease];
}

static MPSGraphTensor* chainViewOperation(ViewCachedGraph* cachedGraph,
                                          const IntArrayRef& size,
                                          const IntArrayRef& stride,
                                          int64_t offset,
                                          const IntArrayRef& base_shape,
                                          bool needsScatter,
                                          MPSGraphTensor* updatesTensor) {
  MPSGraph* mpsGraph = cachedGraph->graph();
  MPSGraphTensor* outputTensor = nil;
  const size_t shape_size = size.size();

  @autoreleasepool {
    std::vector<int32_t> sizeArray(shape_size);
    const int64_t int_max = std::numeric_limits<int32_t>::max();
    for (const auto i : c10::irange(shape_size)) {
      TORCH_CHECK(size[i] <= int_max);
      sizeArray[i] = static_cast<int32_t>(size[i]);
    }
    NSData* shapeData = [NSData dataWithBytes:sizeArray.data() length:shape_size * sizeof(int32_t)];
    MPSGraphTensor* shapeTensor = [mpsGraph constantWithData:shapeData
                                                       shape:@[ [NSNumber numberWithUnsignedInteger:shape_size] ]
                                                    dataType:MPSDataTypeInt32];
    MPSGraphTensor* indicesTensor = nil;
    // create stride Tensors for each rank of the input tensor
    for (int i = 0; i < static_cast<int>(shape_size); i++) {
      MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis:(-i - 1) withShapeTensor:shapeTensor name:nil];
      MPSGraphTensor* strideTensor = cachedGraph->strideTensors[shape_size - i - 1];
      MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor:rangeTensor
                                                              secondaryTensor:strideTensor
                                                                         name:nil];
      if (!indicesTensor) {
        indicesTensor = indexTensor;
      } else {
        indicesTensor = [mpsGraph additionWithPrimaryTensor:indexTensor secondaryTensor:indicesTensor name:nil];
      }
    }

    indicesTensor = [mpsGraph additionWithPrimaryTensor:indicesTensor
                                        secondaryTensor:cachedGraph->storageOffsetTensor
                                                   name:nil];
    MPSGraphTensor* inputTensor = cachedGraph->inputTensor;

    if (!needsScatter) {
      MPSGraphTensor* outputTensor = asStridedLayer_pattern(mpsGraph, inputTensor, shape_size, size, stride, offset);
      if (outputTensor) {
        return outputTensor;
      }
    }

    MPSGraphTensor* reshapedInputTensor = [mpsGraph reshapeTensor:inputTensor withShape:@[ @-1 ] name:nil];
    MPSGraphTensor* reshapedIndicesTensor = [mpsGraph reshapeTensor:indicesTensor withShape:@[ @-1 ] name:nil];
    if (needsScatter) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-method-access"
      MPSGraphTensor* scatteredTensor = [mpsGraph scatterAlongAxis:(NSInteger)0
                                                    withDataTensor:reshapedInputTensor
                                                     updatesTensor:updatesTensor
                                                     indicesTensor:reshapedIndicesTensor
                                                              mode:MPSGraphScatterModeSet
                                                              name:nil];
#pragma clang diagnostic pop
      outputTensor = [mpsGraph reshapeTensor:scatteredTensor withShape:getMPSShape(base_shape) name:nil];
    } else {
      // Call gather to coalesce the needed values. Result will be of same shape as flattened indices tensor
      MPSGraphTensor* gatheredTensor = [mpsGraph gatherWithUpdatesTensor:reshapedInputTensor
                                                           indicesTensor:reshapedIndicesTensor
                                                                    axis:0
                                                         batchDimensions:0
                                                                    name:nil];
      // Reshape the data to desired size
      outputTensor = [mpsGraph reshapeTensor:gatheredTensor withShapeTensor:shapeTensor name:nil];
    }
  }
  return outputTensor;
}

static IntArrayRef updateTensorBaseShape(const Tensor& self) {
  IntArrayRef base_shape = getIMPSAllocator()->getBufferShape(self.storage().data());
  // if there's no base_shape stored in MPSAllocator, then infer it from tensor's size and store it
  if (base_shape.size() == 0) {
    // IntArrayRef wouldn't own the data, so we use a static storage
    static const int64_t shape_1d = 1;
    // self.sizes().size() could be zero
    base_shape = self.sizes().size()
        ? self.sizes()
        : ((self.is_view() && self._base().sizes().size()) ? self._base().sizes() : IntArrayRef(&shape_1d, 1));

    // base_shape will be retained in MPSAllocator until buffer gets recycled
    if (self.storage().data())
      getIMPSAllocator()->setBufferShape(self.storage().data(), base_shape);
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
static ViewCachedGraph* createViewGraph(const Tensor& self,
                                        const Tensor& updates,
                                        IntArrayRef size,
                                        IntArrayRef stride,
                                        int64_t storage_offset,
                                        bool needsScatter) {
  IntArrayRef base_shape = updateTensorBaseShape(self);

  @autoreleasepool {
    string key = getStridedKey(
        self.scalar_type(), updates.scalar_type(), base_shape, size, stride, storage_offset, needsScatter);
    return LookUpOrCreateCachedGraph<ViewCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* updatesTensor = nil;
      // Workaround for MPSShaderLibrary bug in macOS Monterey
      // This is fixed in macOS Ventura
      auto inputType = getMPSScalarType(self.scalar_type());
      if (inputType == MPSDataTypeUInt8 || (inputType == MPSDataTypeBool && !is_macos_13_or_newer())) {
        inputType = MPSDataTypeInt8;
      }

      // Self is the input tensor we are creating view of
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(base_shape));
      newCachedGraph->storageOffsetTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @1 ]);
      for (const auto C10_UNUSED i : c10::irange(size.size())) {
        newCachedGraph->strideTensors.push_back(mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @1 ]));
      }
      if (needsScatter) {
        auto updatesType = getMPSScalarType(updates.scalar_type());
        if (updatesType == MPSDataTypeUInt8 || (updatesType == MPSDataTypeBool && !is_macos_13_or_newer())) {
          updatesType = MPSDataTypeInt8;
        }
        newCachedGraph->updatesTensor = mpsGraphRankedPlaceHolder(mpsGraph, updatesType, getMPSShape(self.numel()));
        updatesTensor = newCachedGraph->updatesTensor;
        if (inputType != updatesType) {
          updatesTensor = [mpsGraph castTensor:updatesTensor toType:inputType name:@"castUpdatesTensor"];
        }
      }
      newCachedGraph->outputTensor =
          chainViewOperation(newCachedGraph, size, stride, storage_offset, base_shape, needsScatter, updatesTensor);
    });
  }
}

static std::string getGatherScatterFunctionName(ScalarType scalarType, int64_t dim, bool needsScatter) {
  std::string kernelName = needsScatter ? "scatter" : "gather";
  return kernelName + "_kernel_" + std::to_string(dim == 0 ? 1 : dim);
}

const std::string& getGatherScatterScalarType(const Tensor& t) {
  auto scalar_type = t.scalar_type();
  static std::unordered_map<c10::ScalarType, std::string> scalarToMetalType = {
      {c10::ScalarType::Float, "float"},
      {c10::ScalarType::Half, "half"},
      {c10::ScalarType::Long, "long"},
      {c10::ScalarType::Int, "int"},
      {c10::ScalarType::Short, "short"},
      {c10::ScalarType::Char, "char"},
      {c10::ScalarType::Byte, "uchar"},
      {c10::ScalarType::Bool, "bool"},
  };

  auto it = scalarToMetalType.find(scalar_type);
  TORCH_CHECK(it != scalarToMetalType.end(), "Unsupported type byte size: ", scalar_type);
  return it->second;
}

static id<MTLLibrary> compileGatherScatterOpsLibrary(id<MTLDevice> device,
                                                     const std::string& dtypeSrc,
                                                     const std::string& dtypeDst,
                                                     bool needsScatter) {
  auto key = std::to_string(needsScatter) + dtypeSrc + dtypeDst;
  static std::unordered_map<std::string, id<MTLLibrary>> _libCache;
  auto it = _libCache.find(key);
  if (it != _libCache.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  auto gatherScatterLib =
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(needsScatter ? SCATTER_OPS_TEMPLATE
                                                                                           : GATHER_OPS_TEMPLATE,
                                                                              dtypeSrc,
                                                                              dtypeDst)
                                                                      .c_str()]
                           options:options
                             error:&error];
  TORCH_CHECK(gatherScatterLib != nil && error == nil,
              "Failed to compile gather-scatter library, error: ",
              [[error description] UTF8String]);
  _libCache[key] = gatherScatterLib;
  return gatherScatterLib;
}

static id<MTLComputePipelineState> getPipelineState(id<MTLDevice> device,
                                                    const std::string& kernel,
                                                    const std::string& dtypeSrc,
                                                    const std::string& dtypeDst,
                                                    bool needsScatter) {
  auto key = kernel + dtypeSrc + dtypeDst;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> _mtlPipelineCache;
  auto it = _mtlPipelineCache.find(key);
  if (it != _mtlPipelineCache.end()) {
    return it->second;
  }

  NSError* error = nil;
  id<MTLLibrary> library = compileGatherScatterOpsLibrary(device, dtypeSrc, dtypeDst, needsScatter);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(func, "Failed to load the Metal Shader function: ", kernel);
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      pso != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  _mtlPipelineCache[key] = pso;
  return pso;
}

Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
  Tensor output = dst;
  if (!dst.has_storage()) {
    output = at::empty(src.sizes(), src.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  }

  if (src.numel() == 0 || output.numel() == 0) {
    return dst;
  }

  if (src.dim() > 5) {
    ViewCachedGraph* cachedGraph =
        createViewGraph(src, dst, src.sizes(), src.strides(), src.storage_offset(), /*needsScatter*/ false);
    return runViewGraph(cachedGraph, src, dst.has_storage() ? dst : output, /*needsScatter*/ false);
  }

  id<MTLBuffer> outputBuffer = dst.has_storage() ? getMTLBufferStorage(dst) : getMTLBufferStorage(output);
  int64_t outputStorageOffset = output.storage_offset() * output.element_size();
  uint32_t numThreads = output.numel();

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
    std::string functionName = getGatherScatterFunctionName(output.scalar_type(), output.dim(), /*needsScatter=*/false);
    id<MTLComputePipelineState> gatherPSO = getPipelineState(MPSDevice::getInstance()->device(),
                                                             functionName,
                                                             getGatherScatterScalarType(src),
                                                             getGatherScatterScalarType(output),
                                                             /*needsScatter=*/false);

    // this function call is a no-op if MPS Profiler is not enabled
    getMPSProfiler().beginProfileKernel(gatherPSO, functionName, {src, output});

    uint32_t kernel_size = src.sizes().size();
    std::vector<uint32_t> src_sizes(kernel_size == 0 ? 1 : kernel_size);
    std::vector<uint32_t> src_strides(kernel_size == 0 ? 1 : kernel_size);

    if (kernel_size == 0) {
      src_sizes[0] = src_strides[0] = 1;
    } else {
      for (const auto i : c10::irange(kernel_size)) {
        src_sizes[i] = (uint32_t)(src.sizes()[i]);
        src_strides[i] = (uint32_t)(src.strides()[i]);
      }
    }

    [computeEncoder setComputePipelineState:gatherPSO];
    [computeEncoder setBuffer:getMTLBufferStorage(src) offset:src.storage_offset() * src.element_size() atIndex:0];
    [computeEncoder setBuffer:outputBuffer offset:outputStorageOffset atIndex:1];
    [computeEncoder setBytes:&src_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:2];
    [computeEncoder setBytes:&src_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
    [computeEncoder setBytes:&numThreads length:sizeof(uint32_t) atIndex:4];

    MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
    NSUInteger threadsPerThreadgroup_ = gatherPSO.maxTotalThreadsPerThreadgroup;
    if (threadsPerThreadgroup_ > numThreads) {
      threadsPerThreadgroup_ = numThreads;
    }

    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroup_, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

    getMPSProfiler().endProfileKernel(gatherPSO);
  });

  return (dst.has_storage()) ? dst : output;
}

Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  if (output.dim() > 5) {
    ViewCachedGraph* cachedGraph = createViewGraph(output.is_complex() ? at::view_as_real(output) : output,
                                                   src,
                                                   output.sizes(),
                                                   output.strides(),
                                                   output.storage_offset(),
                                                   /*needsScatter*/ true);
    return runViewGraph(cachedGraph, src, output, /*needsScatter*/ true);
  }
  if (src.numel() == 0 || output.numel() == 0) {
    return output;
  }

  id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
  id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  uint32_t numThreads = src.numel();
  int64_t outputStorageOffset = output.storage_offset() * output.element_size();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      std::string functionName =
          getGatherScatterFunctionName(output.scalar_type(), output.dim(), /*needsScatter=*/true);
      id<MTLComputePipelineState> scatterPSO = getPipelineState(MPSDevice::getInstance()->device(),
                                                                functionName,
                                                                getGatherScatterScalarType(src),
                                                                getGatherScatterScalarType(output),
                                                                /*needsScatter=*/true);

      getMPSProfiler().beginProfileKernel(scatterPSO, functionName, {src, output});

      uint32_t kernel_size = output.sizes().size();
      std::vector<uint32_t> output_sizes(kernel_size == 0 ? 1 : kernel_size);
      std::vector<uint32_t> output_strides(kernel_size == 0 ? 1 : kernel_size);

      if (kernel_size == 0) {
        output_sizes[0] = output_strides[0] = 1;
      } else {
        for (const auto i : c10::irange(kernel_size)) {
          output_sizes[i] = (uint32_t)(output.sizes()[i]);
          output_strides[i] = (uint32_t)(output.strides()[i]);
        }
      }

      [computeEncoder setComputePipelineState:scatterPSO];
      [computeEncoder setBuffer:sourceBuffer offset:src.storage_offset() * src.element_size() atIndex:0];
      [computeEncoder setBuffer:outputBuffer offset:outputStorageOffset atIndex:1];
      [computeEncoder setBytes:&output_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:2];
      [computeEncoder setBytes:&output_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
      [computeEncoder setBytes:&numThreads length:sizeof(uint32_t) atIndex:4];

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      NSUInteger threadsPerThreadgroup_ = scatterPSO.maxTotalThreadsPerThreadgroup;
      if (threadsPerThreadgroup_ > numThreads) {
        threadsPerThreadgroup_ = numThreads;
      }

      MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroup_, 1, 1);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];

      getMPSProfiler().endProfileKernel(scatterPSO);
    }
  });

  return output;
}

} // namespace mps

// implementation of as_strided() op
Tensor as_strided_tensorimpl_mps(const Tensor& self,
                                 IntArrayRef size,
                                 IntArrayRef stride,
                                 c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result =
      detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // creating the view graph will be deferred until gatherViewTensor() or scatterViewTensor() are called.
  // In as_strided, we just update the base shape of the buffer in order to retrieve it later
  // when we create/run the view graph.
  IntArrayRef base_shape = mps::updateTensorBaseShape(self);
  TORCH_INTERNAL_ASSERT(
      base_shape.size() > 0, "Failed to update the base shape of tensor's buffer at ", self.storage().data());

  return result;
}

} // namespace at::native
