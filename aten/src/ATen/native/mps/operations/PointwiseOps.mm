//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#endif
namespace at::native {
// scope the MPS's internal methods to not expose them to at::native
namespace mps {

void addc_mul_div_out_mps(const Tensor& self,
                          const Tensor& tensor1,
                          const Tensor& tensor2,
                          const Scalar& value_opt, // default value = 1.0
                          const Tensor& output,
                          const bool is_div,
                          const string op_name) {
  if (value_opt.toDouble() == 0.0) {
    output.copy_(self);
    return;
  }

  if (output.numel() == 0) {
    return;
  }

  MPSStream* mpsStream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor *firstTensor = nil, *secondTensor = nil, *valueTensor = nil;
  };

  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, tensor1, tensor2});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      ScalarType common_dtype =
          c10::promoteTypes(self.scalar_type(), c10::promoteTypes(tensor1.scalar_type(), tensor2.scalar_type()));
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      newCachedGraph->firstTensor = mpsGraphRankedPlaceHolder(mpsGraph, tensor1);
      newCachedGraph->secondTensor = mpsGraphRankedPlaceHolder(mpsGraph, tensor2);
      newCachedGraph->valueTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(self.scalar_type()), @[ @1 ]);

      // the tensor to be optionally multiplied by value_scalar
      MPSGraphTensor* multiplicandTensor = nil;
      auto firstTensor = castMPSTensor(mpsGraph, newCachedGraph->firstTensor, common_dtype);
      auto secondTensor = castMPSTensor(mpsGraph, newCachedGraph->secondTensor, common_dtype);
      if (is_div) {
        multiplicandTensor = [mpsGraph divisionWithPrimaryTensor:firstTensor secondaryTensor:secondTensor name:nil];
      } else {
        multiplicandTensor = [mpsGraph multiplicationWithPrimaryTensor:firstTensor
                                                       secondaryTensor:secondTensor
                                                                  name:nil];
      }
      // the tensor to be added to input_tensor
      MPSGraphTensor* addendTensor =
          [mpsGraph multiplicationWithPrimaryTensor:multiplicandTensor
                                    secondaryTensor:castMPSTensor(mpsGraph, newCachedGraph->valueTensor, common_dtype)
                                               name:nil];
      auto outputTensor =
          [mpsGraph additionWithPrimaryTensor:castMPSTensor(mpsGraph, newCachedGraph->inputTensor, common_dtype)
                              secondaryTensor:addendTensor
                                         name:nil];
      newCachedGraph->outputTensor = castMPSTensor(mpsGraph, outputTensor, output.scalar_type());
    });

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

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
  }
}

} // namespace mps

// APIs exposed to at::native scope
TORCH_IMPL_FUNC(addcmul_out_mps)
(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, const Tensor& output) {
  mps::addc_mul_div_out_mps(self, tensor1, tensor2, value, output, false, "addcmul_out_mps");
}

TORCH_IMPL_FUNC(addcdiv_out_mps)
(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, const Tensor& output) {
  mps::addc_mul_div_out_mps(self, tensor1, tensor2, value, output, true, "addcdiv_out_mps");
}

} // namespace at::native
