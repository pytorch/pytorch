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

          MPSDataType self_dtype = getMPSScalarType(self.scalar_type());
          MPSGraphTensor* outputTensor;
          if (self_dtype == MPSDataTypeBool) {
            MPSGraphTensor* inputTensor;
            if (value.toDouble()) {
              // TODO: Simply using value.toDouble() (1.0 for True) does not work!
              //       Results in outputTensor having value of 255,
              //       which displays as "False" in Python frontend! Whats going on...?
              //       Am I simply using constantWithScalar incorrectly or is there a bug in MPSGraph?
              //       dataType argument did not matter, tested float32 and int32,64
              inputTensor = [mpsGraph constantWithScalar:1.1 shape:input_shape dataType:MPSDataTypeFloat32];
            } else {
              inputTensor = [mpsGraph constantWithScalar:0.0 shape:input_shape dataType:MPSDataTypeFloat32];
            }
            outputTensor = [mpsGraph castTensor:inputTensor toType:MPSDataTypeBool name:@"castToBool"];
          } else {
            // TODO: constantWithScalar output is incorrect for large integers because
            //       it only accepts double scalars and furthermore MPS only supports single precision...
            //       therefore bottlenecked by float32 precision even for ints, test:
            //       >>> torch.tensor(16777217, dtype=torch.float32, device="mps")
            //       tensor([16777216.], device='mps:0')
            //       >>> torch.full((1,), 16777217, dtype=torch.int32, device="mps")
            //       tensor([16777216], device='mps:0', dtype=torch.int32)
            //       The first one is expected while the second one is not. On CPU it works, what also works is
            //       torch.tensor(16777217, device="mps"), which I think goes through CPU first and then
            //       copies over to the MPS device.
            MPSGraphTensor* inputTensor = [mpsGraph constantWithScalar:value.toDouble()
                                                                 shape:input_shape
                                                              dataType:self_dtype];
            outputTensor = [mpsGraph identityWithTensor:inputTensor name:nil];
          }
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
