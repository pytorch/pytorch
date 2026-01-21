#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mode_native.h>
#include <ATen/ops/sort.h>
#endif

namespace at::native {
namespace mps {

// Mode implementation for MPS
// Algorithm:
// 1. Sort the tensor along the dimension (with indices)
// 2. Create a mask for where consecutive elements differ
// 3. Use cumsum to count run lengths
// 4. Find the maximum run length and return the corresponding value/index

static void mode_kernel_mps_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {

  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t slice_size = ensure_nonempty_size(self, dim);

  // Resize output value, index Tensors to appropriate sizes
  TORCH_CHECK(0 <= dim && static_cast<size_t>(dim) < self_sizes.size());
  self_sizes[dim] = 1;

  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }

  at::native::resize_output(values, self_sizes);
  at::native::resize_output(indices, self_sizes);

  // Handle slice_size == 1 case
  if (slice_size == 1) {
    values.copy_(self);
    indices.fill_(0);
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }
    return;
  }

  // For MPS, we use sort + run-length encoding to find the mode
  // First, transpose to make dim the last dimension for easier processing
  auto transposed = self.transpose(dim, ndim - 1);
  auto contiguous = transposed.contiguous();

  // Sort along the last dimension and get indices
  auto [sorted_values, sorted_indices] = at::sort(contiguous, -1);

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* sortedValuesTensor = nil;
    MPSGraphTensor* sortedIndicesTensor = nil;
    MPSGraphTensor* outputValuesTensor = nil;
    MPSGraphTensor* outputIndicesTensor = nil;
  };

  @autoreleasepool {
    MPSShape* sorted_shape = getMPSShape(sorted_values);
    NSString* ns_shape_key = [[sorted_shape valueForKey:@"description"] componentsJoinedByString:@","];
    std::string key = std::string("mode:") + [ns_shape_key UTF8String] + ":" + getMPSTypeString(self);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSDataType dataType = getMPSDataType(sorted_values);
      MPSDataType indicesType = getMPSDataType(sorted_indices);

      newCachedGraph->sortedValuesTensor = mpsGraphRankedPlaceHolder(mpsGraph, dataType, sorted_shape);
      newCachedGraph->sortedIndicesTensor = mpsGraphRankedPlaceHolder(mpsGraph, indicesType, sorted_shape);

      MPSGraphTensor* sortedVals = newCachedGraph->sortedValuesTensor;
      MPSGraphTensor* sortedInds = newCachedGraph->sortedIndicesTensor;

      // Cast to float32 if needed for comparisons
      MPSDataType compareType = dataType;
      if (dataType != MPSDataTypeInt32 && dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16) {
        compareType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
        sortedVals = [mpsGraph castTensor:sortedVals toType:compareType name:@"castSortedVals"];
      }

      NSInteger lastDim = (NSInteger)[sorted_shape count] - 1;
      int64_t sliceLen = slice_size;

      // Create mask where consecutive elements differ (along last dimension)
      // frontNMinusOne = sorted[..., :-1]
      // backNMinusOne = sorted[..., 1:]
      MPSGraphTensor* frontNMinusOne = [mpsGraph sliceTensor:sortedVals
                                                   dimension:lastDim
                                                       start:0
                                                      length:sliceLen - 1
                                                        name:nil];
      MPSGraphTensor* backNMinusOne = [mpsGraph sliceTensor:sortedVals
                                                  dimension:lastDim
                                                      start:1
                                                     length:sliceLen - 1
                                                       name:nil];

      // notEqual gives 1 where values change, 0 where they're the same
      MPSGraphTensor* notEqual = [mpsGraph notEqualWithPrimaryTensor:backNMinusOne
                                                     secondaryTensor:frontNMinusOne
                                                                name:nil];
      // Cast to int for cumsum
      MPSGraphTensor* changesMask = [mpsGraph castTensor:notEqual toType:MPSDataTypeInt32 name:@"changesMask"];

      // Prepend a 0 to mark the start of the first run
      NSMutableArray<NSNumber*>* zeroShape = [NSMutableArray array];
      for (NSUInteger i = 0; i < [sorted_shape count] - 1; i++) {
        [zeroShape addObject:sorted_shape[i]];
      }
      [zeroShape addObject:@1];

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0 shape:zeroShape dataType:MPSDataTypeInt32];
      MPSGraphTensor* changesWithZero = [mpsGraph concatTensors:@[ zeroTensor, changesMask ]
                                                      dimension:lastDim
                                                           name:nil];

      // cumsum gives run IDs (elements with same value have same ID)
      MPSGraphTensor* runIds = [mpsGraph cumulativeSumWithTensor:changesWithZero axis:lastDim name:nil];

      // Compute run lengths using scatter_add with runIds
      // First, create a tensor of ones
      MPSGraphTensor* ones = [mpsGraph constantWithScalar:1 shape:sorted_shape dataType:MPSDataTypeInt32];

      // Scatter add the ones to compute counts for each run
      // But we need to know the max number of runs (= slice_size)
      NSMutableArray<NSNumber*>* countShape = [NSMutableArray array];
      for (NSUInteger i = 0; i < [sorted_shape count] - 1; i++) {
        [countShape addObject:sorted_shape[i]];
      }
      [countShape addObject:@(sliceLen)];

      // Scatter add ones to runIds positions
      MPSGraphTensor* runCounts = [mpsGraph scatterWithUpdatesTensor:ones
                                                       indicesTensor:runIds
                                                               shape:countShape
                                                                axis:lastDim
                                                                mode:MPSGraphScatterModeAdd
                                                                name:nil];

      // For each position, get its run's count
      MPSGraphTensor* positionCounts = [mpsGraph gatherWithUpdatesTensor:runCounts
                                                           indicesTensor:runIds
                                                                    axis:lastDim
                                                              batchDims:lastDim
                                                                    name:nil];

      // Find position with maximum count and correct tie-breaking (LAST occurrence)
      // PyTorch's mode returns the last occurrence of the mode value.
      // Among positions with maximal count, select the one with the largest original index.
      MPSGraphTensor* maxCounts = [mpsGraph reductionMaximumWithTensor:positionCounts
                                                                  axis:lastDim
                                                                  name:nil];
      MPSGraphTensor* maxCountMask = [mpsGraph equalWithPrimaryTensor:positionCounts
                                                      secondaryTensor:maxCounts
                                                                 name:nil];
      // Use original indices for tie-breaking: among positions with maximal count,
      // select the one with the largest original index (last occurrence).
      MPSGraphTensor* indicesAsFloat = [mpsGraph castTensor:sortedInds
                                                     toType:compareType
                                                       name:nil];
      MPSGraphTensor* negInfTensor = [mpsGraph constantWithScalar:-INFINITY
                                                         dataType:compareType
                                                            shape:sorted_shape];
      MPSGraphTensor* maskedIndices = [mpsGraph selectWithPredicateTensor:maxCountMask
                                                               trueTensor:indicesAsFloat
                                                              falseTensor:negInfTensor
                                                                     name:nil];
      // Argmax over masked original indices now picks the last occurrence
      // (largest original index) among positions with maximal count.
      MPSGraphTensor* maxCountPos = [mpsGraph reductionArgMaximumWithTensor:maskedIndices axis:lastDim name:nil];

      // Get the mode value using gather
      MPSGraphTensor* modeValues = [mpsGraph gatherWithUpdatesTensor:sortedVals
                                                       indicesTensor:maxCountPos
                                                                axis:lastDim
                                                          batchDims:lastDim
                                                                name:nil];

      // Get the original index using gather
      MPSGraphTensor* modeIndices = [mpsGraph gatherWithUpdatesTensor:sortedInds
                                                        indicesTensor:maxCountPos
                                                                 axis:lastDim
                                                           batchDims:lastDim
                                                                 name:nil];

      // Cast back if we casted for comparison
      if (dataType != compareType) {
        modeValues = [mpsGraph castTensor:modeValues toType:dataType name:@"castBack"];
      }

      newCachedGraph->outputValuesTensor = modeValues;
      newCachedGraph->outputIndicesTensor = [mpsGraph castTensor:modeIndices toType:MPSDataTypeInt64 name:@"castIndices"];
    });

    // Prepare output tensors with correct shape for the transposed result
    auto values_transposed = values.transpose(dim, ndim - 1);
    auto indices_transposed = indices.transpose(dim, ndim - 1);

    Placeholder sortedValsPlaceholder = Placeholder(cachedGraph->sortedValuesTensor, sorted_values);
    Placeholder sortedIndsPlaceholder = Placeholder(cachedGraph->sortedIndicesTensor, sorted_indices);
    Placeholder valuesPlaceholder = Placeholder(cachedGraph->outputValuesTensor, values_transposed);
    Placeholder indicesPlaceholder = Placeholder(cachedGraph->outputIndicesTensor, indices_transposed);

    auto feeds = dictionaryFromPlaceholders(sortedValsPlaceholder, sortedIndsPlaceholder);
    auto results = dictionaryFromPlaceholders(valuesPlaceholder, indicesPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

} // namespace mps

// Register dispatch for mode_stub
void mode_kernel_mps(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  mps::mode_kernel_mps_impl(values, indices, self, dim, keepdim);
}

REGISTER_DISPATCH(mode_stub, &mode_kernel_mps)

} // namespace at::native
