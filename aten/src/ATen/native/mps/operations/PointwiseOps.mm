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
  if (&output != &self) {
    output.resize_(output.sizes());
  }

  if(output.numel() == 0) {
    return output;
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *firstTensor = nil, *secondTensor = nil, *valueTensor = nil;
  };
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, tensor1, tensor2}, false);

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
            newCachedGraph->valueTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(self.scalar_type()));

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
            MPSGraphTensor *addendTensor = [mpsGraph multiplicationWithPrimaryTensor:multiplicandTensor
                                                      secondaryTensor:newCachedGraph->valueTensor
                                                                name:nil];
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
    MPSScalar value_scalar = getMPSScalar(value_opt, self.scalar_type());

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      tensor1Placeholder.getMPSGraphTensor() : tensor1Placeholder.getMPSGraphTensorData(),
      tensor2Placeholder.getMPSGraphTensor() : tensor2Placeholder.getMPSGraphTensorData(),
      cachedGraph->valueTensor : getMPSGraphTensorFromScalar(mpsStream, value_scalar),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
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
