//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

using namespace at::mps;

namespace at {
namespace native {

Tensor& fill_scalar_mps_impl(Tensor& self, const Scalar& value) {
  using namespace mps;

  if (self.numel() == 0) {
    return self;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache *cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    MPSShape* input_shape = getMPSShape(self);
    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "fill_scalar_mps_impl:" + getMPSTypeString(self.scalar_type())
                                         + ":" + string([ns_shape_key UTF8String])
                                         + ":" + to_string(value.toDouble());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = [mpsGraph constantWithScalar:value.toDouble()
                                                               shape:input_shape
                                                            dataType:getMPSScalarType(self.scalar_type())];
          MPSGraphTensor* outputTensor = [mpsGraph identityWithTensor:inputTensor
                                                                 name:nil];

          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, self);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = nil;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return self;
}

Tensor& zero_mps_(Tensor& self) {
  return at::native::fill_scalar_mps_impl(self, 0.0f);
}

Tensor& fill_scalar_mps(Tensor& self, const Scalar& value) {
  return at::native::fill_scalar_mps_impl(self, value);
}

Tensor& fill_tensor_mps_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return at::native::fill_scalar_mps_impl(self, value.item());
}
} // namespace native
} // namespace at
