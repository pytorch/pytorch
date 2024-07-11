//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Resize.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique2.h>
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#endif

namespace at::native {
namespace mps {

struct UniqueCachedGraph : public MPSCachedGraph {
  UniqueCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
  MPSGraphTensor* inverseIndicesTensor_ = nil;
  MPSGraphTensor* countsTensor_ = nil;
  MPSGraphTensor* lengthTensor_ = nil;
};

static std::string getUniqueKey(const ScalarType& dtype,
                                const IntArrayRef& base_shape,
                                const bool return_inverse,
                                const bool return_counts,
                                const bool consecutive,
                                c10::optional<int64_t> dimOpt) {
  return "_unique2_mps:" + getMPSTypeString(dtype) + "[" + getArrayRefString(base_shape) + "]:[" +
      (dimOpt.has_value() ? std::to_string(dimOpt.value()) : "None") + "]:[" + std::to_string(return_inverse) + "]:[" +
      std::to_string(return_counts) + "]:[" + std::to_string(consecutive) + "]";
}

// dim arg not supported when non consecutive, ie sorted
static std::array<MPSGraphTensor*, 4> buildUniqueGraph(const Tensor& self,
                                                       UniqueCachedGraph* uniqueGraph,
                                                       const bool return_inverse,
                                                       const bool return_counts,
                                                       const bool consecutive,
                                                       c10::optional<int64_t> dimOpt) {
  int64_t dim = dimOpt.has_value() ? maybe_wrap_dim(dimOpt.value(), self.dim()) : 0;

  MPSGraph* graph = uniqueGraph->graph();
  MPSGraphTensor* inputTensor = uniqueGraph->inputTensor_;
  MPSShape* shape = [inputTensor shape];
  MPSShape* destShape = shape;
  uint64_t length = [shape[dim] unsignedIntValue];
  MPSDataType dataType = [inputTensor dataType];

  MPSGraphTensor* resultTensor = nil;
  MPSGraphTensor* inverseIndicesTensor = nil;
  MPSGraphTensor* countTensor = nil;
  MPSGraphTensor* lengthTensor = nil;

  const bool needsFlatten = !(dimOpt.has_value() || [shape count] == 1);
  if (needsFlatten) {
    inputTensor = [graph reshapeTensor:inputTensor withShape:@[ @-1 ] name:nil];
    length = 1;
    for (const auto i : c10::irange([shape count])) {
      if (c10::mul_overflows(length, [shape[i] unsignedIntValue], &length)) {
        TORCH_CHECK(false, "RuntimeError: Tensor size overflow");
      }
    }

    destShape = @[ [NSNumber numberWithUnsignedInteger:length] ];
  }

  if (length <= 1) {
    // Trivial case, only 1 element everything is unique
    resultTensor = inputTensor;
    lengthTensor = [graph constantWithScalar:0.0f dataType:MPSDataTypeInt32];
    if (return_inverse) {
      inverseIndicesTensor = [graph constantWithScalar:0.0f dataType:MPSDataTypeInt32];
    }
    if (return_counts) {
      countTensor = [graph constantWithScalar:1.0f dataType:MPSDataTypeInt32];
    }
    return {resultTensor, inverseIndicesTensor, countTensor, lengthTensor};
  }

  // #issue 104398441 sortWithTensor only supports following types, cast if necessary
  if (dataType != MPSDataTypeInt32 && dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16) {
    dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
    inputTensor = [graph castTensor:inputTensor toType:dataType name:@"castInputTensor"];
  }

  MPSGraphTensor* sortedInput = nil;
  if (consecutive) {
    sortedInput = inputTensor;
  } else {
    sortedInput = [graph sortWithTensor:inputTensor axis:0 name:nil];
  }

  MPSGraphTensor* frontNMinusOne = [graph sliceTensor:sortedInput dimension:dim start:0 length:length - 1 name:nil];
  MPSGraphTensor* backNMinusOne = [graph sliceTensor:sortedInput dimension:dim start:1 length:length - 1 name:nil];
  MPSGraphTensor* notEqualToPreviousElement = [graph notEqualWithPrimaryTensor:backNMinusOne
                                                               secondaryTensor:frontNMinusOne
                                                                          name:nil];
  MPSGraphTensor* mask = [graph castTensor:notEqualToPreviousElement toType:MPSDataTypeInt32 name:@"castMaskTensor"];

  // If comparing tensors, not scalars, check if entire tensor matches previous element using reductionOr over tensor
  if (dimOpt.has_value() && [shape count] != 1) {
    NSMutableArray* axes = [[NSMutableArray alloc] initWithCapacity:[shape count] - 1];
    for (const auto axis : c10::irange([shape count])) {
      if (static_cast<decltype(dim)>(axis) != dim) {
        [axes addObject:[NSNumber numberWithUnsignedInteger:axis]];
      }
    }
    mask = [graph reductionOrWithTensor:mask axes:axes name:nil];
    mask = [graph squeezeTensor:mask axes:axes name:nil];
    [axes release];
  }

  MPSGraphTensor* scannedIndices = [graph cumulativeSumWithTensor:mask axis:0 name:nil];
  lengthTensor = [graph sliceTensor:scannedIndices dimension:0 start:length - 2 length:1 name:nil];

  MPSGraphTensor* minusOneTensor = [graph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];
  MPSGraphTensor* maskedIndices = [graph selectWithPredicateTensor:mask
                                               truePredicateTensor:scannedIndices
                                              falsePredicateTensor:minusOneTensor
                                                              name:nil];

  MPSGraphTensor* zeroTensor = [graph constantWithScalar:0.0f shape:@[ @1 ] dataType:MPSDataTypeInt32];
  MPSGraphTensor* maskedIndicesWithHead = [graph concatTensors:@[ zeroTensor, maskedIndices ] dimension:0 name:nil];
  MPSGraphTensor* scannedIndicesWithHead = [graph concatTensors:@[ zeroTensor, scannedIndices ] dimension:0 name:nil];

  resultTensor = [graph scatterWithUpdatesTensor:sortedInput
                                   indicesTensor:maskedIndicesWithHead
                                           shape:destShape
                                            axis:dim
                                            mode:MPSGraphScatterModeSet
                                            name:nil];
  // Cast back if necessary
  if ([uniqueGraph->inputTensor_ dataType] != dataType) {
    resultTensor = [graph castTensor:resultTensor toType:[uniqueGraph->inputTensor_ dataType] name:@"castResultTensor"];
  }

  // Compute optional returned tensors if requested
  if (return_inverse) {
    MPSGraphTensor* argSortedInput = nil;
    if (consecutive)
      argSortedInput = [graph coordinateAlongAxis:0
                                        withShape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                             name:nil];
    else
      argSortedInput = [graph argSortWithTensor:inputTensor axis:0 name:nil];
    inverseIndicesTensor = [graph scatterWithUpdatesTensor:scannedIndicesWithHead
                                             indicesTensor:argSortedInput
                                                     shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                                      axis:0
                                                      mode:MPSGraphScatterModeAdd
                                                      name:nil];
    if (needsFlatten)
      inverseIndicesTensor = [graph reshapeTensor:inverseIndicesTensor withShape:shape name:nil];
  }

  if (return_counts) {
    MPSGraphTensor* unitTensor = [graph constantWithScalar:1.0f
                                                     shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                                  dataType:MPSDataTypeInt32];
    countTensor = [graph scatterWithUpdatesTensor:unitTensor
                                    indicesTensor:scannedIndicesWithHead
                                            shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                             axis:0
                                             mode:MPSGraphScatterModeAdd
                                             name:nil];
  }

  return {resultTensor, inverseIndicesTensor, countTensor, lengthTensor};
}

static UniqueCachedGraph* getUniqueGraph(const Tensor& self,
                                         const bool return_inverse,
                                         const bool return_counts,
                                         const bool consecutive,
                                         c10::optional<int64_t> dim) {
  @autoreleasepool {
    string key = getUniqueKey(self.scalar_type(), self.sizes(), return_inverse, return_counts, consecutive, dim);
    return LookUpOrCreateCachedGraph<UniqueCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(self), getMPSShape(self));
      auto outputTensors = buildUniqueGraph(self, newCachedGraph, return_inverse, return_counts, consecutive, dim);

      newCachedGraph->outputTensor_ = outputTensors[0];
      newCachedGraph->inverseIndicesTensor_ = outputTensors[1];
      newCachedGraph->countsTensor_ = outputTensors[2];
      newCachedGraph->lengthTensor_ = outputTensors[3];
    });
  }
}

static void runUniqueGraph(UniqueCachedGraph* uniqueGraph,
                           const Tensor& input,
                           Tensor& output,
                           Tensor& inverse_indices,
                           Tensor& counts,
                           Tensor& length,
                           bool return_inverse,
                           bool return_counts) {
  Placeholder inputPlaceholder = Placeholder(uniqueGraph->inputTensor_, input);
  auto feeds = dictionaryFromPlaceholders(inputPlaceholder);

  NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [NSMutableDictionary dictionary];
  Placeholder outputPlaceholder = Placeholder(uniqueGraph->outputTensor_, output);
  Placeholder lengthPlaceholder = Placeholder(uniqueGraph->lengthTensor_, length);
  [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
  [results setObject:lengthPlaceholder.getMPSGraphTensorData() forKey:lengthPlaceholder.getMPSGraphTensor()];
  if (return_inverse) {
    Placeholder inverseIndicesPlaceholder = Placeholder(uniqueGraph->inverseIndicesTensor_, inverse_indices);
    [results setObject:inverseIndicesPlaceholder.getMPSGraphTensorData()
                forKey:inverseIndicesPlaceholder.getMPSGraphTensor()];
  }
  if (return_counts) {
    Placeholder countsPlaceholder = Placeholder(uniqueGraph->countsTensor_, counts);
    [results setObject:countsPlaceholder.getMPSGraphTensorData() forKey:countsPlaceholder.getMPSGraphTensor()];
  }

  // Run the graph
  MPSStream* stream = getCurrentMPSStream();
  runMPSGraph(stream, uniqueGraph->graph(), feeds, results);
}

} // namespace mps

static std::tuple<Tensor, Tensor, Tensor> _unique_impl_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive,
                                                           c10::optional<int64_t> dimOpt) {
  const Tensor& input = self.contiguous();

  // get flat output size
  int64_t totalElems = c10::multiply_integers(input.sizes());

  IntArrayRef outputShape = IntArrayRef(totalElems);
  IntArrayRef inverseIndicesShape = input.sizes();
  IntArrayRef countsShape = IntArrayRef(totalElems);
  int64_t dim = dimOpt.has_value() ? maybe_wrap_dim(dimOpt.value(), self.dim()) : 0;

  if (dimOpt.has_value()) {
    outputShape = input.sizes();
    inverseIndicesShape = IntArrayRef(input.sizes()[dim]);
    countsShape = IntArrayRef(input.sizes()[dim]);
  }
  if (!return_inverse)
    inverseIndicesShape = {};
  if (!return_counts)
    countsShape = {};

  Tensor output = at::empty(outputShape, input.scalar_type(), c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  Tensor inverse_indices =
      at::empty(inverseIndicesShape, ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  Tensor counts = at::empty(countsShape, ScalarType::Long, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  Tensor length = at::empty({1}, ScalarType::Int, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);

  if (input.numel() == 0) {
    return std::make_tuple(output, inverse_indices, counts);
  }

  mps::UniqueCachedGraph* uniqueGraph = mps::getUniqueGraph(input, return_inverse, return_counts, consecutive, dimOpt);
  mps::runUniqueGraph(uniqueGraph, input, output, inverse_indices, counts, length, return_inverse, return_counts);

  int64_t lengthScalar = length.item<int64_t>() + 1; // length actually holds max index, add 1
  if (output.sizes().size() != 0) {
    output = at::slice(output, dim, 0, lengthScalar);
  }
  if (return_counts)
    counts = at::slice(counts, 0, 0, lengthScalar);

  return std::make_tuple(output, inverse_indices, counts);
}

static std::tuple<Tensor, Tensor, Tensor> castToMPS(std::tuple<Tensor, Tensor, Tensor> out) {
  return std::make_tuple(std::get<0>(out).to("mps"), std::get<1>(out).to("mps"), std::get<2>(out).to("mps"));
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_mps(const Tensor& self,
                                                          const bool return_inverse,
                                                          const bool return_counts,
                                                          c10::optional<int64_t> dim) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: unique_consecutive op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return castToMPS(at::unique_consecutive(self.to("cpu"), return_inverse, return_counts, dim));
  }

  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_mps(const Tensor& self,
                                                              int64_t dim,
                                                              const bool return_inverse,
                                                              const bool return_counts) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: unique_dim_consecutive op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return castToMPS(at::unique_dim_consecutive(self.to("cpu"), dim, return_inverse, return_counts));
  }

  return _unique_impl_mps(self, return_inverse, return_counts, true, c10::make_optional((int64_t)dim));
}

std::tuple<Tensor, Tensor, Tensor> _unique2_mps(const Tensor& self,
                                                const bool sorted,
                                                const bool return_inverse,
                                                const bool return_counts) {
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: _unique2 op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");
    return castToMPS(at::_unique2(self.to("cpu"), sorted, return_inverse, return_counts));
  }

  return _unique_impl_mps(self, return_inverse, return_counts, false, c10::nullopt);
}

} // namespace at::native
