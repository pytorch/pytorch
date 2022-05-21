//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>

namespace at {
namespace native {
// scope the MPS's internal methods to not expose them to at::native
namespace mps {

Tensor& addc_mul_div_out_mps(const Tensor& self,
                             const Tensor& tensor1,
                             const Tensor& tensor2,
                             const Scalar& value_opt, // default value = 1.0
                             Tensor& output,
                             const bool is_div,
                             const string op_name)
{
  using scalar_t = double;
  scalar_t value_scalar = value_opt.to<scalar_t>();
  if (&output != &self) {
    output.resize_(output.sizes());
  }
  TORCH_CHECK(output.is_mps());

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *firstTensor = nil, *secondTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = op_name + to_string(value_scalar)
                         + getTensorsStringKey({self, tensor1, tensor2})+ ":"
                         + getMPSTypeString(value_opt.type());

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
        MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph* newCachedGraph = nil;
          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
            newCachedGraph->firstTensor = mpsGraphRankedPlaceHolder(mpsGraph, tensor1);
            newCachedGraph->secondTensor = mpsGraphRankedPlaceHolder(mpsGraph, tensor2);

            // the tensor to be optionally multiplied by value_scalar
            MPSGraphTensor *multiplicandTensor = nil;
            if (is_div) {
              multiplicandTensor = [mpsGraph divisionWithPrimaryTensor:newCachedGraph->firstTensor
                                                       secondaryTensor:newCachedGraph->secondTensor
                                                                  name:nil];
            } else {
              multiplicandTensor = [mpsGraph multiplicationWithPrimaryTensor:newCachedGraph->firstTensor
                                                             secondaryTensor:newCachedGraph->secondTensor
                                                                        name:nil];
            }
            // the tensor to be added to input_tensor
            MPSGraphTensor *addendTensor = multiplicandTensor;
            // if value_scalar is 1.0, then we don't bother adding another multiply to graph
            if (value_scalar != 1.0) {
              MPSGraphTensor* valueTensor = [mpsGraph constantWithScalar:value_scalar
                                                                dataType:getMPSScalarType(value_opt.type())];
              addendTensor = [mpsGraph multiplicationWithPrimaryTensor:multiplicandTensor
                                                       secondaryTensor:valueTensor
                                                                  name:nil];
            }
            newCachedGraph->outputTensor = [mpsGraph additionWithPrimaryTensor:newCachedGraph->inputTensor
                                                               secondaryTensor:addendTensor
                                                                          name:nil];
          }
          return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    // Inputs as placeholders
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor, self);
    Placeholder tensor1Placeholder = Placeholder(cachedGraph->firstTensor, tensor1);
    Placeholder tensor2Placeholder = Placeholder(cachedGraph->secondTensor, tensor2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);

    // Create dictionary of inputs and outputs
    // Utility to dump out graph : [mpsGraph dump];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      tensor1Placeholder.getMPSGraphTensor() : tensor1Placeholder.getMPSGraphTensorData(),
      tensor2Placeholder.getMPSGraphTensor() : tensor2Placeholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }

  return output;
}

} // namespace mps

// APIs exposed to at::native scope
TORCH_IMPL_FUNC(addcmul_out_mps)
(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, const Tensor& output)
{
  mps::addc_mul_div_out_mps(self, tensor1, tensor2, value, const_cast<Tensor&>(output), false, "addcmul_out_mps");
}

TORCH_IMPL_FUNC(addcdiv_out_mps)
(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, const Tensor& output)
{
  mps::addc_mul_div_out_mps(self, tensor1, tensor2, value, const_cast<Tensor&>(output), true, "addcdiv_out_mps");
}

} // namespace native
} // namespace at
