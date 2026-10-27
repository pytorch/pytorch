//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>
#include <ATen/native/Gelu.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/relu_native.h>
#endif

using namespace at::mps;

namespace at::native {
using namespace mps;

TORCH_IMPL_FUNC(log_softmax_backward_mps_out)
(const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, const Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kLong, "MPS doesn't know how to do exponent_i64");
  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(grad_output.scalar_type()),
                              "log_softmax for complex is not supported for MPS");
  TORCH_CHECK_NOT_IMPLEMENTED(grad_output.scalar_type() != kBool, "log_softmax for bool is not supported for MPS");
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  if (output.numel() == 0) {
    return;
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "log_softmax_backward_mps_out:" + getMPSTypeString(grad_output) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_output));
      MPSGraphTensor* outputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(output));

      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:outputTensor name:nil];
      MPSGraphTensor* sumTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor axis:dim name:nil];
      MPSGraphTensor* multiplicationTensor = [mpsGraph multiplicationWithPrimaryTensor:expTensor
                                                                       secondaryTensor:sumTensor
                                                                                  name:nil];
      MPSGraphTensor* resultTensor = [mpsGraph subtractionWithPrimaryTensor:gradOutputTensor
                                                            secondaryTensor:multiplicationTensor
                                                                       name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->gradInputTensor_ = resultTensor;
    });

    Placeholder gradPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder resultPlaceholder = Placeholder(cachedGraph->gradInputTensor_, out);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradPlaceholder, outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, resultPlaceholder);
  }
}

TORCH_IMPL_FUNC(gelu_out_mps)(const Tensor& self, std::string_view approximate, const Tensor& output) {
  GeluKernel(kMPS, *this, get_gelutype_enum(approximate));
}

TORCH_IMPL_FUNC(gelu_backward_out_mps)
(const Tensor& grad, const Tensor& self, std::string_view approximate, const Tensor& grad_input) {
  GeluBackwardKernel(kMPS, *this, get_gelutype_enum(approximate));
}

std::tuple<Tensor, Tensor> prelu_backward_mps(const Tensor& grad_output, const Tensor& self, const Tensor& weight_) {
  using namespace mps;

  Tensor grad_input = at::empty_like(self, self.suggest_memory_format());
  Tensor weight_grad = at::empty_like(self, at::MemoryFormat::Contiguous);
  if (grad_output.numel() == 0) {
    return std::tuple<Tensor, Tensor>{grad_input, weight_grad};
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* weightedGradTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "prelu_backward_mps:" + getTensorsStringKey({grad_output, self, weight_});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* weightedGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:weightTensor
                                                                           secondaryTensor:gradOutputTensor
                                                                                      name:nil];
      MPSGraphTensor* inputGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                        secondaryTensor:gradOutputTensor
                                                                                   name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:zeroTensor
                                                                          name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:gradOutputTensor
                                                    falsePredicateTensor:weightedGradOutputTensor
                                                                    name:nil];
      MPSGraphTensor* weightedGradTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                           truePredicateTensor:zeroTensor
                                                          falsePredicateTensor:inputGradOutputTensor
                                                                          name:nil];
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->weightedGradTensor_ = weightedGradTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->outputTensor_, grad_input);
    Placeholder weightedGradPlaceholder = Placeholder(cachedGraph->weightedGradTensor_, weight_grad);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder, weightPlaceholder);
    auto results = dictionaryFromPlaceholders(gradInputPlaceholder, weightedGradPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
  return std::tuple<Tensor, Tensor>{grad_input, weight_grad};
}

} // namespace at::native
