#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include "c10/util/Optional.h"

namespace at {
namespace native {

Tensor& eye_out_mps(int64_t n, Tensor& result) {
  // the default value of `m` equals to `n`
  return eye_out_mps(n, n, result);
}

Tensor& eye_out_mps(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});
  result.zero_();

  if(result.numel() == 0)
    return result;

  // Get stream
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "eye_out_mps:" + getTensorsStringKey({result});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          // Initialize graph
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);
          MPSGraphTensor* onesTensor = [mpsGraph constantWithScalar:1.0f
                                                              shape:getMPSShape(result)
                                                           dataType:getMPSDataType(result.scalar_type())];

          MPSGraphTensor* outputTensor = [mpsGraph bandPartWithTensor:onesTensor
                                                             numLower:0
                                                             numUpper:0
                                                                 name:nil];
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    // Create placeholders
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = nil;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}


}
}