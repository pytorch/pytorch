//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(triu_mps_out)
(const Tensor& self,
 int64_t k,
 const Tensor &output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "triu_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1
                                                               dataType:MPSDataTypeInt32];

          if(k > 0) {
            MPSGraphTensor* diagMinusOneTensor = [mpsGraph constantWithScalar:(k-1)
                                                                     dataType:MPSDataTypeInt32];
            MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                             numLowerTensor:minusOneTensor
                                                             numUpperTensor:diagMinusOneTensor
                                                                       name:nil];
            outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                  secondaryTensor:complementTensor
                                                             name:nil];
          }
          else {
            MPSGraphTensor* minusDiagTensor = [mpsGraph constantWithScalar:(-k)
                                                                  dataType:MPSDataTypeInt32];
            outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                         numLowerTensor:minusDiagTensor
                                         numUpperTensor:minusOneTensor
                                                   name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

TORCH_IMPL_FUNC(tril_mps_out)
(const Tensor& self,
 int64_t k,
 const Tensor &output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "tril_mps_out" + mps::getTensorsStringKey({self}) + ":" + std::to_string(k);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = nil;

          MPSGraphTensor* minusOneTensor = [mpsGraph constantWithScalar:-1
                                                               dataType:MPSDataTypeInt32];

          if(k >= 0) {
            MPSGraphTensor* diagTensor = [mpsGraph constantWithScalar:k
                                                             dataType:MPSDataTypeInt32];
            outputTensor = [mpsGraph bandPartWithTensor:inputTensor
                                         numLowerTensor:minusOneTensor
                                         numUpperTensor:diagTensor
                                                   name:nil];
          }
          else {
            MPSGraphTensor* negDiagMinusOneTensor = [mpsGraph constantWithScalar:(-k-1)
                                                                        dataType:MPSDataTypeInt32];
            MPSGraphTensor* complementTensor = [mpsGraph bandPartWithTensor:inputTensor
                                                             numLowerTensor:negDiagMinusOneTensor
                                                             numUpperTensor:minusOneTensor
                                                                       name:nil];
            outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                  secondaryTensor:complementTensor
                                                             name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

}

} // namespace native
} // namespace at
