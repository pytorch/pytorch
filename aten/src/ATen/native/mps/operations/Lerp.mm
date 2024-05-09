#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/lerp_native.h>
#endif

namespace at::native {
TORCH_IMPL_FUNC(lerp_Tensor_mps)(const Tensor& self, const Tensor& end, const Tensor& weight, const Tensor& out) {
  TORCH_CHECK(out.is_mps());
  std::array<TensorArg, 4> args{{{out, "out", 0}, {self, "self", 1}, {end, "end", 2}, {weight, "weight", 3}}};
  checkAllSameGPU(__func__, args);
  using namespace mps;
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* endTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };
  @autoreleasepool {
    string key = "lerp_Tensor_mps" + getTensorsStringKey({self, end, weight});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto graph) {
      auto selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      auto endTensor = mpsGraphRankedPlaceHolder(mpsGraph, end);
      auto weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
      auto distance = [mpsGraph subtractionWithPrimaryTensor:endTensor secondaryTensor:selfTensor name:nil];
      auto weighedDistance = [mpsGraph multiplicationWithPrimaryTensor:weightTensor secondaryTensor:distance name:nil];
      auto output = [mpsGraph additionWithPrimaryTensor:selfTensor secondaryTensor:weighedDistance name:nil];
      graph->selfTensor_ = selfTensor;
      graph->endTensor_ = endTensor;
      graph->weightTensor_ = weightTensor;
      graph->outputTensor_ = output;
    });
    auto selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    auto endPlaceholder = Placeholder(cachedGraph->endTensor_, end);
    auto weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, endPlaceholder, weightPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

} // namespace at::native
