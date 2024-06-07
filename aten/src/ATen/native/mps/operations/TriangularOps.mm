//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#endif

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at::native {

TORCH_IMPL_FUNC(triu_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "triu_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* outputTensor = nil;

      MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt32];

      if (k > 0) {
        MPSGraphTensor* diagMinusOneTensor = [mpsGraph constantWithScalar:(k - 1) dataType:MPSDataTypeInt32];
        MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                         numLowerTensor:minusOneTensor
                                                         numUpperTensor:diagMinusOneTensor
                                                                   name:nil];
        outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor secondaryTensor:complementTensor name:nil];
      } else {
        MPSGraphTensor* minusDiagTensor = [mpsGraph constantWithScalar:(-k) dataType:MPSDataTypeInt32];
        outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                     numLowerTensor:minusDiagTensor
                                     numUpperTensor:minusOneTensor
                                               name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(tril_mps_out)
(const Tensor& self, int64_t k, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "tril_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* outputTensor = nil;

      MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt32];

      if (k >= 0) {
        MPSGraphTensor* diagTensor = [mpsGraph constantWithScalar:k dataType:MPSDataTypeInt32];
        outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                     numLowerTensor:minusOneTensor
                                     numUpperTensor:diagTensor
                                               name:nil];
      } else {
        MPSGraphTensor* negDiagMinusOneTensor = [mpsGraph constantWithScalar:(-k - 1) dataType:MPSDataTypeInt32];
        MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                         numLowerTensor:negDiagMinusOneTensor
                                                         numUpperTensor:minusOneTensor
                                                                   name:nil];
        outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor secondaryTensor:complementTensor name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

} // namespace at::native
