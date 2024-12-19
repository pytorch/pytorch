#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/angle_native.h>
#endif

namespace at::native {

Tensor angle_mps(const Tensor& self) {
  const auto float_type = c10::toRealValueType(self.scalar_type());
  Tensor result = at::empty({0}, self.options().dtype(float_type));
  return angle_out_mps(self, result);
}

using namespace mps;

Tensor& angle_out_mps(const Tensor& self, Tensor& result) {
  result.resize_(self.sizes());
  result.zero_();

  // Handle empty outputs
  if (result.numel() == 0)
    return result;

  // Get MPS stream
  MPSStream* stream = getCurrentMPSStream();

  auto outputDataType = c10::toRealValueType(self.scalar_type());
  // Derive from MPSCachedGraph
  // This structure is used to cache an MPSGraph with certain keys, so that we don't have to compile the same MPSGraph
  // time and time again for the same operation The keys of this structure are based on the inputs and outputs needed
  // for the operation here, we don't have any input tensors, just an output tensor.
  // If the operator to be added is unary or binary, instead of creating a new CachedGraph struct yourself, please
  // consider using `MPSUnaryCachedGraph` or `MPSBinaryCachedGraph` and their corresponding Grad versions in
  // `OperationUtils.h`.
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
  };

  @autoreleasepool {
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types
    // etc match the earlier created MPSGraph
    string key = "angle_out_mps:" + getTensorsStringKey({self, result});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // Here we can call the MPSGraph API needed to execute the operation.
      // The API details can be found here:
      // https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph
      MPSGraphTensor* real_part = [mpsGraph realPartOfTensor:newCachedGraph->inputTensor name:nil];
      MPSGraphTensor* imag_part = [mpsGraph imaginaryPartOfTensor:newCachedGraph->inputTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph atan2WithPrimaryTensor:imag_part secondaryTensor:real_part name:nil];

      if ([outputTensor dataType] != getMPSDataType(outputDataType)) {
        outputTensor = castMPSTensor(mpsGraph, outputTensor, outputDataType);
      }
      newCachedGraph->outputTensor = outputTensor;
    });

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, result);

    // Create dictionary of inputs/feeds and outputs/results
    // In this case, there are no inputs, so the feeds are nil
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    };
    auto results = dictionaryFromPlaceholders(outputPlaceholder);

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

} // namespace at::native
