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

// Steps to add op for MPS backend:
// Register the op in aten/src/ATen/native/native_functions.yaml with the "MPS" dispatch key
// Define the function interface for the MPS backend based on the interface as defined in the corresponding CPU/CUDA counterparts
// Check in the native_functions.yaml if the function is structured or non-structured
// Add boiler-plate error checking code as present in the CPU/CUDA counterparts

Tensor& eye_out_mps(int64_t n, int64_t m, Tensor& result) {

  // This is one example of boiler-plate error checking, taking after CPU/CUDA counterparts
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});
  result.zero_();

  // Handle empty outputs
  if(result.numel() == 0)
    return result;

  // Get MPS stream
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();

  // Derive from MPSCachedGraph
  // This structure is used to cache an MPSGraph with certain keys, so that we don't have to compile the same MPSGraph time and time again for the same operation
  // The keys of this structure are based on the inputs and outputs needed for the operation
  // Here, we don't have any input tensors, just an output tensor
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types etc match the earlier created MPSGraph
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

          // Here we can call the MPSGraph API needed to execute the operation. 
          // The API details can be found here: https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph
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

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    // Create dictionary of inputs/feeds and outputs/results
    // In this case, there are no inputs, so the feeds are nil
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