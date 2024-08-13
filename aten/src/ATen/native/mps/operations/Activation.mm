//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/elu_backward_native.h>
#include <ATen/ops/elu_native.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/glu_native.h>
#include <ATen/ops/hardsigmoid_backward_native.h>
#include <ATen/ops/hardsigmoid_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/hardswish_native.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/sigmoid_backward_native.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/softshrink_backward_native.h>
#include <ATen/ops/softshrink_native.h>
#include <ATen/ops/tanh_backward_native.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>
#endif

using namespace at::mps;

namespace at::native {

Tensor relu_mps(const Tensor& self) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor output = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  if (output.numel() == 0) {
    return output;
  }

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    string key = "relu" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      // passing selector of reLUWithTensor on the mpsGraph object
      MPSGraphTensor* outputTensor = [mpsGraph reLUWithTensor:inputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return output;
}

Tensor& relu_mps_(Tensor& self) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return self;
  }
  // Inplace relu
  Tensor& output = self;
  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor out;
  if (executeGatherOp) {
    out = at::empty_like(self, MemoryFormat::Contiguous);
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "relu_" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      // passing selector of reLUWithTensor on the mpsGraph object
      MPSGraphTensor* outputTensor = [mpsGraph reLUWithTensor:inputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, executeGatherOp ? out : output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
    if (executeGatherOp) {
      output.copy_(out);
    }
  }

  return output;
}

TORCH_IMPL_FUNC(leaky_relu_out_mps)(const Tensor& self, const Scalar& negative_slope, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  TORCH_CHECK(output.is_mps());

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor output_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    string key = "leaky_relu" + getTensorsStringKey({self}) + ":" + std::to_string(negative_slope.to<double>());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* negSlopeTensor = [mpsGraph constantWithScalar:negative_slope.to<double>()
                                                              shape:@[ @1 ]
                                                           dataType:getMPSDataType(self)];
      MPSGraphTensor* negSlopeMulXTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                     secondaryTensor:negSlopeTensor
                                                                                name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph maximumWithPrimaryTensor:negSlopeMulXTensor
                                                        secondaryTensor:inputTensor
                                                                   name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, executeGatherOp ? output_ : output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  if (executeGatherOp) {
    output.copy_(output_);
  }
}

TORCH_IMPL_FUNC(leaky_relu_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& self,
 const Scalar& negative_slope,
 bool self_is_result,
 const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(output.is_mps());

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  Tensor output_ = at::empty_like(self, self.suggest_memory_format());

  @autoreleasepool {
    string key = "leaky_relu_backward" + getTensorsStringKey({self, grad_output}) + ":" +
        std::to_string(negative_slope.to<double>());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* negSlopeTensor = [mpsGraph constantWithScalar:negative_slope.to<double>()
                                                              shape:@[ @1 ]
                                                           dataType:getMPSScalarType(self)];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f shape:@[ @1 ] dataType:getMPSScalarType(self)];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:zeroTensor
                                                                          name:nil];
      MPSGraphTensor* gradientsMulNegSlopeTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                                             secondaryTensor:negSlopeTensor
                                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                        truePredicateTensor:gradOutputTensor
                                                       falsePredicateTensor:gradientsMulNegSlopeTensor
                                                                       name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, output_);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  output.copy_(output_);
}

TORCH_IMPL_FUNC(log_softmax_mps_out)
(const Tensor& self, const int64_t dim, const bool half_to_float, const Tensor& out) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return;
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "log_softmax_mps_out" + getTensorsStringKey({self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* maximumsTensor = [mpsGraph reductionMaximumWithTensor:inputTensor axis:dim name:nil];
      MPSGraphTensor* inputTensorSubMax = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:maximumsTensor
                                                                            name:nil];
      MPSGraphTensor* exponentTensor = [mpsGraph exponentWithTensor:inputTensorSubMax name:nil];

      MPSGraphTensor* exponentTensorReduced = [mpsGraph reductionSumWithTensor:exponentTensor axis:dim name:nil];

      MPSGraphTensor* logSumExpTensor = [mpsGraph logarithmWithTensor:exponentTensorReduced name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph subtractionWithPrimaryTensor:inputTensorSubMax
                                                            secondaryTensor:logSumExpTensor
                                                                       name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(log_softmax_backward_mps_out)
(const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, const Tensor& out) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  if (output.numel() == 0) {
    return;
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "log_softmax_backward_mps_out:" + getMPSTypeString(grad_output) + ":" + std::to_string(dim);
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

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_mps(const Tensor& self, Tensor& output, Tensor& buffer) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (self.numel() == 0) {
    return std::forward_as_tuple(output, buffer);
  }

  output.resize_as_(self);

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor output_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    string key = "log_sigmoid_forward_out:" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* minTensor = [mpsGraph minimumWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
      MPSGraphTensor* absInputTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
      MPSGraphTensor* negAbsInputTensor = [mpsGraph negativeWithTensor:absInputTensor name:nil];
      MPSGraphTensor* expNegAbsInputTensor = [mpsGraph exponentWithTensor:negAbsInputTensor name:nil];
      MPSGraphTensor* outputTensor = at::native::mps::log1p(mpsGraph, expNegAbsInputTensor);
      outputTensor = [mpsGraph subtractionWithPrimaryTensor:minTensor secondaryTensor:outputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, executeGatherOp ? output_ : output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (executeGatherOp) {
    output.copy_(output_);
  }
  return std::forward_as_tuple(output, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_mps(const Tensor& self) {
  auto output = at::empty_like(self);
  auto buffer = at::empty({0}, self.options());
  log_sigmoid_forward_out_mps(self, output, buffer);
  return std::make_tuple(output, buffer);
}

Tensor& log_sigmoid_backward_mps_out(const Tensor& grad_output,
                                     const Tensor& self,
                                     const Tensor& buffer,
                                     Tensor& grad_input) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  if (self.numel() == 0) {
    return grad_input;
  }

  grad_input.resize_as_(self);

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor grad_input_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    string key = "log_sigmoid_backward_out:" + getTensorsStringKey({self, grad_output});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* negOneTensor = [mpsGraph constantWithScalar:-1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* inputNegPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                    secondaryTensor:zeroTensor
                                                                               name:nil];
      MPSGraphTensor* maxDerivativeTensor = [mpsGraph selectWithPredicateTensor:inputNegPredicateTensor
                                                            truePredicateTensor:oneTensor
                                                           falsePredicateTensor:zeroTensor
                                                                           name:nil];
      MPSGraphTensor* signTensor = [mpsGraph selectWithPredicateTensor:inputNegPredicateTensor
                                                   truePredicateTensor:oneTensor
                                                  falsePredicateTensor:negOneTensor
                                                                  name:nil];
      MPSGraphTensor* absInputTensor = [mpsGraph absoluteWithTensor:inputTensor name:nil];
      MPSGraphTensor* negAbsInputTensor = [mpsGraph negativeWithTensor:absInputTensor name:nil];
      MPSGraphTensor* expNegAbsInputTensor = [mpsGraph exponentWithTensor:negAbsInputTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:expNegAbsInputTensor
                                                         secondaryTensor:oneTensor
                                                                    name:nil];
      outputTensor = [mpsGraph divisionWithPrimaryTensor:expNegAbsInputTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:signTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph subtractionWithPrimaryTensor:maxDerivativeTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor secondaryTensor:outputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->gradInputTensor_, executeGatherOp ? grad_input_ : grad_input, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (executeGatherOp) {
    grad_input.copy_(grad_input_);
  }
  return grad_input;
}

Tensor log_sigmoid_backward_mps(const Tensor& grad_output, const Tensor& self, const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  log_sigmoid_backward_mps_out(grad_output, self, buffer, grad_input);
  return grad_input;
}

TORCH_IMPL_FUNC(sigmoid_backward_out_mps)(const Tensor& grad_output, const Tensor& output, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(grad_input.is_mps());

  if (grad_output.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "sigmoid_backward_out_mps:" + getMPSTypeString(grad_output);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_output));
      MPSGraphTensor* outputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(output));

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* oneMinusSigmoidTensor = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                                     secondaryTensor:outputTensor
                                                                                name:nil];
      MPSGraphTensor* timesTensor = [mpsGraph multiplicationWithPrimaryTensor:oneMinusSigmoidTensor
                                                              secondaryTensor:outputTensor
                                                                         name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                                  secondaryTensor:timesTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

TORCH_IMPL_FUNC(tanh_backward_out_mps)(const Tensor& grad_output, const Tensor& output, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(grad_input.is_mps());

  if (grad_output.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "tanh_backward_out_mps:" + getMPSTypeString(grad_output);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(grad_output));
      MPSGraphTensor* outputTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(output));

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* tanh2Tensor = [mpsGraph squareWithTensor:outputTensor name:nil];
      MPSGraphTensor* oneMinusTanh2Tensor = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                                   secondaryTensor:tanh2Tensor
                                                                              name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                                  secondaryTensor:oneMinusTanh2Tensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

TORCH_IMPL_FUNC(threshold_out_mps)
(const Tensor& self, const Scalar& threshold, const Scalar& value, const Tensor& result) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  TORCH_CHECK(self.is_mps());

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "threshold_out_mps" + getTensorsStringKey({self}) + ":" + std::to_string(threshold.to<double>()) +
        ":" + std::to_string(value.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* thresholdTensor = [mpsGraph constantWithScalar:threshold.to<double>()
                                                               shape:@[ @1 ]
                                                            dataType:getMPSDataType(self)];

      MPSGraphTensor* valueTensor = [mpsGraph constantWithScalar:value.to<double>()
                                                           shape:@[ @1 ]
                                                        dataType:getMPSDataType(self)];

      // x > threshold
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:thresholdTensor
                                                                          name:nil];

      // result = (self > threshold) ? self : value
      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:inputTensor
                                                    falsePredicateTensor:valueTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(threshold_backward_out_mps)
(const Tensor& grad, const Tensor& self, const Scalar& threshold, const Tensor& gradInput) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(self.is_mps());
  TORCH_CHECK(grad.is_mps());

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key =
        "threshold_backward_out_mps" + getTensorsStringKey({self, grad}) + ":" + std::to_string(threshold.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad);

      MPSGraphTensor* thresholdTensor = [mpsGraph constantWithScalar:threshold.to<double>()
                                                               shape:@[ @1 ]
                                                            dataType:getMPSDataType(self)];

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];

      // x > threshold
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:thresholdTensor
                                                                          name:nil];

      // result = (self > threshold) ? grad : zeroTensor
      MPSGraphTensor* gradInputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                        truePredicateTensor:gradTensor
                                                       falsePredicateTensor:zeroTensor
                                                                       name:nil];

      newCachedGraph->gradOutputTensor_ = gradTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, gradInput);

    auto feeds = dictionaryFromPlaceholders(gradPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static MPSGraphTensor* normcdf(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // (1.0f + erf(x*SQRT1_2)) * 0.5f * x;
  auto dataType = [inputTensor dataType];
  const float SQRT1_2 = 0.707106781186547524400844362104849039f;
  MPSGraphTensor* sqrt1_2 = [mpsGraph constantWithScalar:SQRT1_2 shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];

  MPSGraphTensor* erfTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:sqrt1_2 name:nil];
  erfTensor = [mpsGraph erfWithTensor:erfTensor name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:onef name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:halff name:nil];

  return erfTensor;
}

static MPSGraphTensor* tanh(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // 0.5 * x * (1 + text{Tanh}(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  auto dataType = [inputTensor dataType];
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715f;
  MPSGraphTensor* betaf = [mpsGraph constantWithScalar:kBeta shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* kappaf = [mpsGraph constantWithScalar:kKappa shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* erfTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                        secondaryTensor:inputTensor
                                                                   name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:inputTensor name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:kappaf name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:inputTensor name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:betaf name:nil];
  erfTensor = [mpsGraph tanhWithTensor:erfTensor name:nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor:erfTensor secondaryTensor:onef name:nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor:erfTensor secondaryTensor:halff name:nil];

  return erfTensor;
}

TORCH_IMPL_FUNC(gelu_out_mps)(const Tensor& self, c10::string_view approximate, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;
  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(c10::isFloatingType(self.scalar_type()), "GELU is only implemented for floating types");

  // Empty output
  if (output.numel() == 0)
    return;

  auto approximate_type = get_gelutype_enum(approximate);
  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor output_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    const auto key = "gelu_out_mps" + getTensorsStringKey({self}) + ":" + gelutype_to_string(approximate_type);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));

      MPSGraphTensor* outputTensor = nil;
      if (approximate_type == GeluType::Tanh) {
        outputTensor = tanh(mpsGraph, inputTensor);
      } else {
        outputTensor = normcdf(mpsGraph, inputTensor);
      }
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:outputTensor secondaryTensor:inputTensor name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, executeGatherOp ? output_ : output, nil, false);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  if (executeGatherOp) {
    output.copy_(output_);
  }
}

TORCH_IMPL_FUNC(gelu_backward_out_mps)
(const Tensor& grad, const Tensor& self, c10::string_view approximate, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  // Empty output
  if (self.numel() == 0) {
    return;
  }

  Tensor grad_input_ = at::empty_like(self, self.suggest_memory_format());

  auto approximate_type = get_gelutype_enum(approximate);
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    const auto key =
        "gelu_backward_out_mps" + getTensorsStringKey({self, grad}) + ":" + gelutype_to_string(approximate_type);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      auto dataType = getMPSDataType(self);

      MPSGraphTensor* gradTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad), getMPSShape(grad));
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, dataType, getMPSShape(self));
      MPSGraphTensor* outputTensor = nil;
      if (approximate_type == GeluType::Tanh) {
        constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * (0.5f);
        constexpr float kKappa = 0.044715f;
        MPSGraphTensor* betaf = [mpsGraph constantWithScalar:kBeta shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* kappaf = [mpsGraph constantWithScalar:kKappa shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* threef = [mpsGraph constantWithScalar:3.0f shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* x_sq = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                         secondaryTensor:inputTensor
                                                                    name:nil];
        MPSGraphTensor* x_cube = [mpsGraph multiplicationWithPrimaryTensor:x_sq secondaryTensor:inputTensor name:nil];
        MPSGraphTensor* inner = [mpsGraph multiplicationWithPrimaryTensor:kappaf secondaryTensor:x_cube name:nil];
        inner = [mpsGraph additionWithPrimaryTensor:inner secondaryTensor:inputTensor name:nil];
        inner = [mpsGraph multiplicationWithPrimaryTensor:betaf secondaryTensor:inner name:nil];
        MPSGraphTensor* tanhInner = [mpsGraph tanhWithTensor:inner name:nil];
        MPSGraphTensor* left = [mpsGraph multiplicationWithPrimaryTensor:halff secondaryTensor:inputTensor name:nil];
        MPSGraphTensor* right = [mpsGraph additionWithPrimaryTensor:onef secondaryTensor:tanhInner name:nil];
        MPSGraphTensor* left_derivative = [mpsGraph multiplicationWithPrimaryTensor:halff
                                                                    secondaryTensor:right
                                                                               name:nil];
        MPSGraphTensor* tanh_derivative = [mpsGraph multiplicationWithPrimaryTensor:tanhInner
                                                                    secondaryTensor:tanhInner
                                                                               name:nil];
        tanh_derivative = [mpsGraph subtractionWithPrimaryTensor:onef secondaryTensor:tanh_derivative name:nil];
        MPSGraphTensor* inner_derivative = [mpsGraph multiplicationWithPrimaryTensor:threef
                                                                     secondaryTensor:kappaf
                                                                                name:nil];
        inner_derivative = [mpsGraph multiplicationWithPrimaryTensor:inner_derivative secondaryTensor:x_sq name:nil];
        inner_derivative = [mpsGraph additionWithPrimaryTensor:inner_derivative secondaryTensor:onef name:nil];
        inner_derivative = [mpsGraph multiplicationWithPrimaryTensor:betaf secondaryTensor:inner_derivative name:nil];
        MPSGraphTensor* right_derivative = [mpsGraph multiplicationWithPrimaryTensor:left
                                                                     secondaryTensor:tanh_derivative
                                                                                name:nil];
        right_derivative = [mpsGraph multiplicationWithPrimaryTensor:right_derivative
                                                     secondaryTensor:inner_derivative
                                                                name:nil];
        outputTensor = [mpsGraph additionWithPrimaryTensor:left_derivative secondaryTensor:right_derivative name:nil];
        outputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor secondaryTensor:outputTensor name:nil];
      } else {
        constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * (0.5);
        MPSGraphTensor* halff = [mpsGraph constantWithScalar:-0.5f shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* betaf = [mpsGraph constantWithScalar:kBeta shape:@[ @1 ] dataType:dataType];
        MPSGraphTensor* cdf = normcdf(mpsGraph, inputTensor);
        MPSGraphTensor* pdfMul = [mpsGraph squareWithTensor:inputTensor name:nil];
        pdfMul = [mpsGraph multiplicationWithPrimaryTensor:pdfMul secondaryTensor:halff name:nil];
        pdfMul = [mpsGraph exponentWithTensor:pdfMul name:nil];
        MPSGraphTensor* pdf = [mpsGraph multiplicationWithPrimaryTensor:pdfMul secondaryTensor:betaf name:nil];
        pdf = [mpsGraph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:pdf name:nil];
        pdf = [mpsGraph additionWithPrimaryTensor:pdf secondaryTensor:cdf name:nil];
        outputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor secondaryTensor:pdf name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });

    Placeholder gradPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input_);

    auto feeds = dictionaryFromPlaceholders(gradPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  grad_input.copy_(grad_input_);
}

static void elu_variants_out_mps(const Tensor& self,
                                 const Scalar& alpha,
                                 const Scalar& scale,
                                 const Scalar& input_scale,
                                 const Tensor& result,
                                 string func_name) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  auto resultMemFormat = result.suggest_memory_format();
  bool executeGatherOp = !(self.is_contiguous(resultMemFormat) && result.is_contiguous(resultMemFormat));
  Tensor out;
  if (executeGatherOp) {
    out = at::empty_like(result, MemoryFormat::Contiguous);
  }

  // Empty output
  if (result.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = func_name + ":" + getTensorsStringKey({self}) + ":" + std::to_string(alpha.to<double>()) + ":" +
        std::to_string(scale.to<double>()) + ":" + std::to_string(input_scale.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // scale * (max(0, x) + min(0, alpha * (exp(input_scale * x) - 1) ))

      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.to<double>()
                                                           shape:@[ @1 ]
                                                        dataType:getMPSDataType(self)];

      MPSGraphTensor* inputScaleTensor = [mpsGraph constantWithScalar:input_scale.to<double>()
                                                                shape:@[ @1 ]
                                                             dataType:getMPSDataType(self)];

      MPSGraphTensor* scaleTensor = [mpsGraph constantWithScalar:scale.to<double>()
                                                           shape:@[ @1 ]
                                                        dataType:getMPSDataType(self)];
      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* scaledInputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                    secondaryTensor:inputScaleTensor
                                                                               name:nil];
      MPSGraphTensor* exponentTensor = [mpsGraph exponentWithTensor:scaledInputTensor name:nil];
      MPSGraphTensor* exponentMinusOneTensor = [mpsGraph subtractionWithPrimaryTensor:exponentTensor
                                                                      secondaryTensor:unitTensor
                                                                                 name:nil];
      MPSGraphTensor* alphaTimesTensor = [mpsGraph multiplicationWithPrimaryTensor:exponentMinusOneTensor
                                                                   secondaryTensor:alphaTensor
                                                                              name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                               secondaryTensor:zeroTensor
                                                                          name:nil];
      MPSGraphTensor* fusedOutput = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                    truePredicateTensor:inputTensor
                                                   falsePredicateTensor:alphaTimesTensor
                                                                   name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph multiplicationWithPrimaryTensor:fusedOutput
                                                               secondaryTensor:scaleTensor
                                                                          name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out.has_storage() ? out : result, nil, false);
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
    if (out.has_storage()) {
      result.copy_(out);
    }
  }
}

// scale * (max(0, x) + min(0, alpha * (exp(input_scale * x) - 1) ))
TORCH_IMPL_FUNC(elu_out_mps)
(const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, const Tensor& result) {
  elu_variants_out_mps(self, alpha, scale, input_scale, result, "elu_out_mps");
}

TORCH_IMPL_FUNC(elu_backward_out_mps)
(const Tensor& grad_output,
 const Scalar& alpha,
 const Scalar& scale,
 const Scalar& input_scale,
 bool is_result,
 const Tensor& self_or_result,
 const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  auto gradMemFormat = grad_input.suggest_memory_format();
  bool executeGatherOp = !(grad_output.is_contiguous(gradMemFormat) && self_or_result.is_contiguous(gradMemFormat) &&
                           grad_input.is_contiguous(gradMemFormat));
  Tensor out;
  if (executeGatherOp && gradMemFormat == MemoryFormat::ChannelsLast) {
    out = at::empty_like(grad_input, MemoryFormat::Contiguous);
  }

  // Empty output
  if (grad_input.numel() == 0) {
    return;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "elu_backward_out_mps:" + getTensorsStringKey({grad_output, self_or_result}) + ":" +
        std::to_string(alpha.to<double>()) + ":" + std::to_string(scale.to<double>()) + ":" +
        std::to_string(input_scale.to<double>()) + ":" + std::to_string(is_result);

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* selfOrResultTensor = mpsGraphRankedPlaceHolder(mpsGraph, self_or_result);
      MPSGraphTensor* lessThanZeroGradTensor = nil;

      if (is_result) {
        MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.to<double>()
                                                             shape:@[ @1 ]
                                                          dataType:getMPSDataType(grad_output)];
        MPSGraphTensor* resultPlusAlphaTensor = [mpsGraph additionWithPrimaryTensor:selfOrResultTensor
                                                                    secondaryTensor:alphaTensor
                                                                               name:nil];
        auto constMul = scale.to<double>() * input_scale.to<double>();
        MPSGraphTensor* constMulTensor = [mpsGraph constantWithScalar:constMul
                                                                shape:@[ @1 ]
                                                             dataType:getMPSDataType(grad_output)];
        lessThanZeroGradTensor = [mpsGraph multiplicationWithPrimaryTensor:resultPlusAlphaTensor
                                                           secondaryTensor:constMulTensor
                                                                      name:nil];
      } else {
        MPSGraphTensor* inputScaleTensor = [mpsGraph constantWithScalar:input_scale.to<double>()
                                                                  shape:@[ @1 ]
                                                               dataType:getMPSDataType(grad_output)];
        MPSGraphTensor* scaledInputTensor = [mpsGraph multiplicationWithPrimaryTensor:selfOrResultTensor
                                                                      secondaryTensor:inputScaleTensor
                                                                                 name:nil];
        MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:scaledInputTensor name:nil];
        auto constMul = scale.to<double>() * input_scale.to<double>() * alpha.to<double>();
        MPSGraphTensor* constMulTensor = [mpsGraph constantWithScalar:constMul
                                                                shape:@[ @1 ]
                                                             dataType:getMPSDataType(grad_output)];
        lessThanZeroGradTensor = [mpsGraph multiplicationWithPrimaryTensor:expTensor
                                                           secondaryTensor:constMulTensor
                                                                      name:nil];
      }

      MPSGraphTensor* scaleTensor = [mpsGraph constantWithScalar:scale.to<double>()
                                                           shape:@[ @1 ]
                                                        dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:selfOrResultTensor
                                                               secondaryTensor:zeroTensor
                                                                          name:nil];
      MPSGraphTensor* gradTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                   truePredicateTensor:scaleTensor
                                                  falsePredicateTensor:lessThanZeroGradTensor
                                                                  name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = selfOrResultTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output, nil, executeGatherOp);
    Placeholder selfOrResultPlaceholder = Placeholder(cachedGraph->inputTensor_, self_or_result, nil, executeGatherOp);
    Placeholder gradInputPlaceholder =
        Placeholder(cachedGraph->gradInputTensor_, out.has_storage() ? out : grad_input, nil, false);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfOrResultPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
    if (out.has_storage()) {
      grad_input.copy_(out);
    }
  }
}

TORCH_IMPL_FUNC(glu_out_mps)(const Tensor& self, const int64_t dim, const Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(output.is_mps());

  // Empty output
  if (output.numel() == 0)
    return;

  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", nIn);

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "glu_out_mps" + getTensorsStringKey({self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
      NSArray<MPSGraphTensor*>* outputTensorsArray = [mpsGraph splitTensor:inputTensor
                                                                 numSplits:2
                                                                      axis:wrap_dim
                                                                      name:nil];
      MPSGraphTensor* firstHalf = outputTensorsArray[0];
      MPSGraphTensor* secondHalf = [mpsGraph sigmoidWithTensor:outputTensorsArray[1] name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph multiplicationWithPrimaryTensor:firstHalf
                                                               secondaryTensor:secondHalf
                                                                          name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

Tensor& glu_backward_mps_out(const Tensor& grad_output, const Tensor& self, const int64_t dim, Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  // Empty output
  if (grad_input.numel() == 0)
    return grad_input;

  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", nIn);

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "glu_backward_mps_out" + getTensorsStringKey({grad_output, self}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(self), getMPSShape(self));
      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_output), getMPSShape(grad_output));
      NSArray<MPSGraphTensor*>* inputTensorsArray = [mpsGraph splitTensor:inputTensor
                                                                numSplits:2
                                                                     axis:wrap_dim
                                                                     name:nil];

      // first half
      MPSGraphTensor* sigmoidOutputTensor = [mpsGraph sigmoidWithTensor:inputTensorsArray[1] name:nil];
      MPSGraphTensor* firstHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:sigmoidOutputTensor
                                                                        secondaryTensor:gradOutputTensor
                                                                                   name:nil];

      // second half
      MPSGraphTensor* one_val = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* secondHalfOutputTensor = [mpsGraph subtractionWithPrimaryTensor:one_val
                                                                      secondaryTensor:sigmoidOutputTensor
                                                                                 name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:sigmoidOutputTensor
                                                                    name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:inputTensorsArray[0]
                                                                    name:nil];
      secondHalfOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:secondHalfOutputTensor
                                                         secondaryTensor:gradOutputTensor
                                                                    name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph concatTensor:firstHalfOutputTensor
                                                 withTensor:secondHalfOutputTensor
                                                  dimension:wrap_dim
                                                       name:nil];
      newCachedGraph->gradInputTensor_ = outputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
    });

    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
  return grad_input;
}

Tensor glu_backward_mps(const Tensor& grad_output, const Tensor& self, const int64_t dim) {
  Tensor grad_input = at::empty(self.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  grad_input = glu_backward_mps_out(grad_output, self, dim, grad_input);
  return grad_input;
}

TORCH_IMPL_FUNC(softplus_out_mps)
(const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& result) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());
  // Applies the Softplus function :math:`\text{Softplus}(x) = \frac{1}{\beta} *
  // \log(1 + \exp(\beta * x))` element-wise.
  // For numerical stability the implementation reverts to the linear function
  // when :math:`input \times \beta > threshold`.

  // Empty output
  if (result.numel() == 0)
    return;

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* betaTensor_ = nil;
    MPSGraphTensor* thresholdTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();
  MPSScalar beta_scalar = getMPSScalar(beta, self.scalar_type());
  MPSScalar threshold_scalar = getMPSScalar(threshold, self.scalar_type());

  @autoreleasepool {
    string key = "softplus_out_mps:" + getTensorsStringKey({self}) + ":" + std::to_string(beta.to<double>()) + ":" +
        std::to_string(threshold.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* betaTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* thresholdTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* reluTensor = [mpsGraph reLUWithTensor:inputTensor name:nil];

      MPSGraphTensor* reciprocalBetaTensor = [mpsGraph reciprocalWithTensor:betaTensor name:nil];
      MPSGraphTensor* bxTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                           secondaryTensor:betaTensor
                                                                      name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:bxTensor
                                                               secondaryTensor:thresholdTensor
                                                                          name:nil];
      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:bxTensor name:nil];
      MPSGraphTensor* log1pTensor = at::native::mps::log1p(mpsGraph, expTensor);
      MPSGraphTensor* softplusTensor = [mpsGraph multiplicationWithPrimaryTensor:log1pTensor
                                                                 secondaryTensor:reciprocalBetaTensor
                                                                            name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:reluTensor
                                                    falsePredicateTensor:softplusTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->betaTensor_ = betaTensor;
      newCachedGraph->thresholdTensor_ = thresholdTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->betaTensor_ : getMPSGraphTensorFromScalar(stream, beta_scalar),
      cachedGraph->thresholdTensor_ : getMPSGraphTensorFromScalar(stream, threshold_scalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(softplus_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& grad_input) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());

  // Empty output
  if (grad_input.numel() == 0)
    return;

  MPSScalar beta_scalar = getMPSScalar(beta, self.scalar_type());
  MPSScalar threshold_scalar = getMPSScalar(threshold, self.scalar_type());

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* betaTensor_ = nil;
    MPSGraphTensor* thresholdTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "softplus_backward_out_mps:" + getTensorsStringKey({grad_output, self}) + ":" +
        std::to_string(beta.to<double>()) + ":" + std::to_string(threshold.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* betaTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* thresholdTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* bxTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                           secondaryTensor:betaTensor
                                                                      name:nil];
      MPSGraphTensor* expBxTensor = [mpsGraph exponentWithTensor:bxTensor name:nil];
      MPSGraphTensor* unitExpBxTensor = [mpsGraph additionWithPrimaryTensor:expBxTensor
                                                            secondaryTensor:unitTensor
                                                                       name:nil];
      MPSGraphTensor* rTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                          secondaryTensor:expBxTensor
                                                                     name:nil];
      rTensor = [mpsGraph divisionWithPrimaryTensor:rTensor secondaryTensor:unitExpBxTensor name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph greaterThanWithPrimaryTensor:bxTensor
                                                               secondaryTensor:thresholdTensor
                                                                          name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                     truePredicateTensor:gradOutputTensor
                                                    falsePredicateTensor:rTensor
                                                                    name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->betaTensor_ = betaTensor;
      newCachedGraph->thresholdTensor_ = thresholdTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->outputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->betaTensor_ : getMPSGraphTensorFromScalar(stream, beta_scalar),
      cachedGraph->thresholdTensor_ : getMPSGraphTensorFromScalar(stream, threshold_scalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

TORCH_IMPL_FUNC(mish_out_mps)
(const Tensor& self, const Tensor& result) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());

  if (result.numel() == 0)
    return;

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  bool executeGatherOp =
      !(self.is_contiguous(MemoryFormat::Contiguous) || self.is_contiguous(MemoryFormat::ChannelsLast) ||
        self.is_contiguous(MemoryFormat::ChannelsLast3d));
  Tensor result_ = at::empty_like(self, executeGatherOp ? MemoryFormat::Contiguous : MemoryFormat::Preserve);

  @autoreleasepool {
    string key = "mish_out_mps:" + getTensorsStringKey({self});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:inputTensor name:nil];
      MPSGraphTensor* softplusTensor = at::native::mps::log1p(mpsGraph, expTensor);
      MPSGraphTensor* tanhTensor = [mpsGraph tanhWithTensor:softplusTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                               secondaryTensor:tanhTensor
                                                                          name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, executeGatherOp ? result_ : result, nil, false);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  if (executeGatherOp) {
    result.copy_(result_);
  }
}

Tensor mish_backward_mps(const Tensor& grad_output, const Tensor& self) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());

  Tensor grad_input = at::empty_like(self, self.suggest_memory_format());
  if (grad_input.numel() == 0)
    return grad_input;

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "mish_backward_out_mps:" + getTensorsStringKey({grad_output, self});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // grad_input = grad_output * (tanh(softplus(x)) + input * sigmoid(x) * (1 - tanh(softplus(x)) ^ 2)

      MPSGraphTensor* sigmoidTensor = [mpsGraph sigmoidWithTensor:inputTensor name:nil];

      MPSGraphTensor* expTensor = [mpsGraph exponentWithTensor:inputTensor name:nil];
      MPSGraphTensor* softplusTensor = at::native::mps::log1p(mpsGraph, expTensor);
      MPSGraphTensor* tanhSPTensor = [mpsGraph tanhWithTensor:softplusTensor name:nil];

      MPSGraphTensor* tanhSPSquaredTensor = [mpsGraph multiplicationWithPrimaryTensor:tanhSPTensor
                                                                      secondaryTensor:tanhSPTensor
                                                                                 name:nil];
      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* oneMinusTanhSPSquaredTensor = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                                           secondaryTensor:tanhSPSquaredTensor
                                                                                      name:nil];
      MPSGraphTensor* xSigmoidTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:sigmoidTensor
                                                                            name:nil];
      MPSGraphTensor* partialResultTensor = [mpsGraph multiplicationWithPrimaryTensor:xSigmoidTensor
                                                                      secondaryTensor:oneMinusTanhSPSquaredTensor
                                                                                 name:nil];
      MPSGraphTensor* mishGradTensor = [mpsGraph additionWithPrimaryTensor:partialResultTensor
                                                           secondaryTensor:tanhSPTensor
                                                                      name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                                  secondaryTensor:mishGradTensor
                                                                             name:nil];
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
    return grad_input;
  }
}

TORCH_IMPL_FUNC(softshrink_out_mps)
(const Tensor& self, const Scalar& lambd, const Tensor& result) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());

  if (result.numel() == 0)
    return;

  MPSScalar lambd_scalar = getMPSScalar(lambd, self.scalar_type());

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* lambdTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "softshrink_out_mps:" + getTensorsStringKey({self}) + ":" + std::to_string(lambd.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* lambdTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* negativeLambdTensor = [mpsGraph negativeWithTensor:lambdTensor name:nil];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
      MPSGraphTensor* positiveLambdPredicateTensor = [mpsGraph greaterThanOrEqualToWithPrimaryTensor:inputTensor
                                                                                     secondaryTensor:lambdTensor
                                                                                                name:nil];
      MPSGraphTensor* negativeLambdPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                         secondaryTensor:negativeLambdTensor
                                                                                    name:nil];
      MPSGraphTensor* outputTensor =
          [mpsGraph selectWithPredicateTensor:positiveLambdPredicateTensor
                          truePredicateTensor:[mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                     secondaryTensor:lambdTensor
                                                                                name:nil]
                         falsePredicateTensor:zeroTensor
                                         name:nil];
      outputTensor = [mpsGraph selectWithPredicateTensor:negativeLambdPredicateTensor
                                     truePredicateTensor:[mpsGraph additionWithPrimaryTensor:inputTensor
                                                                             secondaryTensor:lambdTensor
                                                                                        name:nil]
                                    falsePredicateTensor:outputTensor
                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->lambdTensor_ = lambdTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->lambdTensor_ : getMPSGraphTensorFromScalar(stream, lambd_scalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void shrink_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& self,
                                    const Scalar& lambd,
                                    const Tensor& grad_input,
                                    std::string op_name) {
  using namespace mps;
  TORCH_CHECK(self.is_mps());

  if (grad_input.numel() == 0)
    return;

  MPSScalar lambd_scalar = getMPSScalar(lambd, self.scalar_type());

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* lambdTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = op_name + ":" + getTensorsStringKey({self}) + ":" + std::to_string(lambd.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* lambdTensor = mpsGraphScalarPlaceHolder(mpsGraph, inputTensor.dataType);

      MPSGraphTensor* negativeLambdTensor = [mpsGraph negativeWithTensor:lambdTensor name:nil];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
      MPSGraphTensor* positiveLambdPredicateTensor = [mpsGraph greaterThanOrEqualToWithPrimaryTensor:inputTensor
                                                                                     secondaryTensor:lambdTensor
                                                                                                name:nil];
      MPSGraphTensor* negativeLambdPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                         secondaryTensor:negativeLambdTensor
                                                                                    name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph selectWithPredicateTensor:positiveLambdPredicateTensor
                                                        truePredicateTensor:gradOutputTensor
                                                       falsePredicateTensor:zeroTensor
                                                                       name:nil];
      gradInputTensor = [mpsGraph selectWithPredicateTensor:negativeLambdPredicateTensor
                                        truePredicateTensor:gradOutputTensor
                                       falsePredicateTensor:gradInputTensor
                                                       name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
      newCachedGraph->lambdTensor_ = lambdTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      cachedGraph->lambdTensor_ : getMPSGraphTensorFromScalar(stream, lambd_scalar),
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
    return;
  }
}

TORCH_IMPL_FUNC(softshrink_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Scalar& lambd, const Tensor& grad_input) {
  return shrink_backward_out_mps(grad_output, self, lambd, grad_input, "softshrink_backward_out_mps");
}

Tensor prelu_mps(const Tensor& self, const Tensor& weight_) {
  using namespace mps;

  Tensor result = at::empty_like(self, self.suggest_memory_format());
  TORCH_INTERNAL_ASSERT(weight_.defined());

  if (result.numel() == 0) {
    return result;
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "prelu_mps:" + getTensorsStringKey({self, weight_});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* reluTensor = [mpsGraph reLUWithTensor:inputTensor name:nil];
      MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                            secondaryTensor:zeroTensor
                                                                       name:nil];
      MPSGraphTensor* weightedTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                                       truePredicateTensor:inputTensor
                                                      falsePredicateTensor:zeroTensor
                                                                      name:nil];
      weightedTensor = [mpsGraph multiplicationWithPrimaryTensor:weightedTensor secondaryTensor:weightTensor name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:reluTensor
                                                         secondaryTensor:weightedTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, weightPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return result;
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
    string key = "prelu_backward_mps:" + getTensorsStringKey({grad_output, self, weight_});

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

TORCH_IMPL_FUNC(silu_out_mps)(const Tensor& self, const Tensor& result) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(self.is_mps());

  // Empty output
  if (result.numel() == 0)
    return;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "silu_out_mps:" + getTensorsStringKey({self});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* negativeInput = [mpsGraph negativeWithTensor:inputTensor name:nil];
      MPSGraphTensor* expNegativeTensor = [mpsGraph exponentWithTensor:negativeInput name:nil];
      MPSGraphTensor* expPlusOneTensor = [mpsGraph additionWithPrimaryTensor:expNegativeTensor
                                                             secondaryTensor:unitTensor
                                                                        name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph divisionWithPrimaryTensor:inputTensor
                                                         secondaryTensor:expPlusOneTensor
                                                                    name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(silu_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(grad_output.is_mps());

  // Empty output
  if (grad_input.numel() == 0)
    return;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "silu_out_backward_mps:" + getTensorsStringKey({grad_output});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* negativeInput = [mpsGraph negativeWithTensor:inputTensor name:nil];
      MPSGraphTensor* expNegativeTensor = [mpsGraph exponentWithTensor:negativeInput name:nil];
      MPSGraphTensor* expPlusOneTensor = [mpsGraph additionWithPrimaryTensor:expNegativeTensor
                                                             secondaryTensor:unitTensor
                                                                        name:nil];
      MPSGraphTensor* sigmoidTensor = [mpsGraph reciprocalWithTensor:expPlusOneTensor name:nil];
      MPSGraphTensor* oneMinusSigmoid = [mpsGraph subtractionWithPrimaryTensor:unitTensor
                                                               secondaryTensor:sigmoidTensor
                                                                          name:nil];
      MPSGraphTensor* inputTimesDiff = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:oneMinusSigmoid
                                                                            name:nil];
      MPSGraphTensor* onePlusTensor = [mpsGraph additionWithPrimaryTensor:unitTensor
                                                          secondaryTensor:inputTimesDiff
                                                                     name:nil];
      MPSGraphTensor* gradTensor = [mpsGraph multiplicationWithPrimaryTensor:sigmoidTensor
                                                             secondaryTensor:onePlusTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
}

TORCH_IMPL_FUNC(hardsigmoid_out_mps)(const Tensor& self, const Tensor& result) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  TORCH_CHECK(self.is_mps());

  // Empty output
  if (result.numel() == 0)
    return;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "hardsigmoid_out_mps:" + getTensorsStringKey({self});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* threeTensor = [mpsGraph constantWithScalar:3.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* sixTensor = [mpsGraph constantWithScalar:6.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* inputPlusThreeTensor = [mpsGraph additionWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:threeTensor
                                                                            name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph clampWithTensor:inputPlusThreeTensor
                                                minValueTensor:zeroTensor
                                                maxValueTensor:sixTensor
                                                          name:nil];
      outputTensor = [mpsGraph divisionWithPrimaryTensor:outputTensor secondaryTensor:sixTensor name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(hardsigmoid_backward_out_mps)
(const Tensor& grad_output, const Tensor& self, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(self.is_mps());

  // Empty output
  if (grad_input.numel() == 0)
    return;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "hardsigmoid_backward_out_mps:" + getTensorsStringKey({self});

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* highTensor = [mpsGraph constantWithScalar:3.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:-3.0 shape:@[ @1 ] dataType:getMPSDataType(self)];
      MPSGraphTensor* oneSixTensor = [mpsGraph constantWithScalar:1.0 / 6.0
                                                            shape:@[ @1 ]
                                                         dataType:getMPSDataType(self)];
      MPSGraphTensor* inputLessThanHighPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                             secondaryTensor:highTensor
                                                                                        name:nil];
      MPSGraphTensor* inputGreaterThanLowPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                                  secondaryTensor:lowTensor
                                                                                             name:nil];
      MPSGraphTensor* inIntervalTensor = [mpsGraph logicalANDWithPrimaryTensor:inputLessThanHighPredicateTensor
                                                               secondaryTensor:inputGreaterThanLowPredicateTensor
                                                                          name:nil];
      MPSGraphTensor* outputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradOutputTensor
                                                               secondaryTensor:oneSixTensor
                                                                          name:nil];

      outputTensor = [mpsGraph selectWithPredicateTensor:inIntervalTensor
                                     truePredicateTensor:outputTensor
                                    falsePredicateTensor:zeroTensor
                                                    name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, gradOutputPlaceholder);
    auto results = dictionaryFromPlaceholders(gradInputPlaceholder);

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

// -------------------------------------------------
// Hardtanh backward

Tensor hardtanh_backward_mps(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max) {
  Tensor grad_input =
      at::empty(grad_output.sizes(), grad_output.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  grad_input = hardtanh_backward_out_mps(grad_output, self, min, max, grad_input);
  return grad_input;
}

// Hardtanh backward
Tensor& hardtanh_backward_out_mps(const Tensor& grad_output,
                                  const Tensor& self,
                                  const Scalar& min,
                                  const Scalar& max,
                                  Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;
  TORCH_CHECK(grad_output.is_mps());

  // Empty output
  if (grad_input.numel() == 0)
    return grad_input;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "hardtanh_backward_out_mps:" + getTensorsStringKey({grad_output}) + ":" +
        std::to_string(min.to<double>()) + ":" + std::to_string(max.to<double>());

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // TODO: Compute gradient
      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* minTensor = [mpsGraph constantWithScalar:min.to<double>()
                                                         shape:@[ @1 ]
                                                      dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* maxTensor = [mpsGraph constantWithScalar:max.to<double>()
                                                         shape:@[ @1 ]
                                                      dataType:getMPSDataType(grad_output)];
      MPSGraphTensor* greaterThanMaxPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                             secondaryTensor:maxTensor
                                                                                        name:nil];
      MPSGraphTensor* lesserThanMinPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                         secondaryTensor:minTensor
                                                                                    name:nil];
      MPSGraphTensor* greaterThanMaxGradTensor = [mpsGraph selectWithPredicateTensor:greaterThanMaxPredicateTensor
                                                                 truePredicateTensor:zeroTensor
                                                                falsePredicateTensor:unitTensor
                                                                                name:nil];
      MPSGraphTensor* lesserThanMinGradTensor = [mpsGraph selectWithPredicateTensor:lesserThanMinPredicateTensor
                                                                truePredicateTensor:zeroTensor
                                                               falsePredicateTensor:unitTensor
                                                                               name:nil];
      MPSGraphTensor* gradTensor = [mpsGraph multiplicationWithPrimaryTensor:greaterThanMaxGradTensor
                                                             secondaryTensor:lesserThanMinGradTensor
                                                                        name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    auto results = dictionaryFromPlaceholders(gradInputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return grad_input;
}

Tensor& hardswish_out_mps(const Tensor& self, Tensor& output) {
  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  if (output.numel() == 0) {
    return output;
  }

  auto resultMemFormat = output.suggest_memory_format();
  bool executeGatherOp = !(self.is_contiguous(resultMemFormat) && output.is_contiguous(resultMemFormat));
  Tensor out;
  if (executeGatherOp && !output.is_contiguous(MemoryFormat::Contiguous)) {
    out = at::empty_like(output, MemoryFormat::Contiguous);
  }

  MPSStream* stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    string key = "hardswish_out_mps" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* threeTensor = [mpsGraph constantWithScalar:3.0f shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* negativeThreeTensor = [mpsGraph constantWithScalar:-3.0f
                                                                   shape:@[ @1 ]
                                                                dataType:getMPSDataType(self)];

      MPSGraphTensor* sixTensor = [mpsGraph constantWithScalar:6.0f shape:@[ @1 ] dataType:getMPSDataType(self)];

      MPSGraphTensor* lessThanMinPredicateTensor = [mpsGraph lessThanOrEqualToWithPrimaryTensor:inputTensor
                                                                                secondaryTensor:negativeThreeTensor
                                                                                           name:nil];

      MPSGraphTensor* lessThanMaxPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                       secondaryTensor:threeTensor
                                                                                  name:nil];

      MPSGraphTensor* inputPlusThreeTensor = [mpsGraph additionWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:threeTensor
                                                                            name:nil];

      MPSGraphTensor* inputDivSixTensor = [mpsGraph divisionWithPrimaryTensor:inputPlusThreeTensor
                                                              secondaryTensor:sixTensor
                                                                         name:nil];

      MPSGraphTensor* weightedTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                                                 secondaryTensor:inputDivSixTensor
                                                                            name:nil];

      MPSGraphTensor* tempTensor = [mpsGraph selectWithPredicateTensor:lessThanMaxPredicateTensor
                                                   truePredicateTensor:weightedTensor
                                                  falsePredicateTensor:inputTensor
                                                                  name:nil];

      MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:lessThanMinPredicateTensor
                                                     truePredicateTensor:zeroTensor
                                                    falsePredicateTensor:tempTensor
                                                                    name:nil];
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, nil, executeGatherOp);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor_, out.has_storage() ? out : output, nil, false);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(selfPlaceholder);
    auto results = dictionaryFromPlaceholders(outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    if (out.has_storage()) {
      output.copy_(out);
    }
  }
  return output;
}

Tensor hardswish_mps(const Tensor& self) {
  using namespace mps;
  Tensor output = at::empty_like(self, self.suggest_memory_format());

  return hardswish_out_mps(self, output);
}

Tensor& hardswish_mps_(Tensor& self) {
  using namespace mps;
  Tensor& output = self;

  return hardswish_out_mps(self, output);
}

Tensor hardswish_backward_mps(const Tensor& grad_output, const Tensor& self) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  Tensor grad_input = at::empty_like(self, self.suggest_memory_format());
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  @autoreleasepool {
    string key = "hardswish_backward_mps" + getTensorsStringKey({self});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];

      MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];

      MPSGraphTensor* threeTensor = [mpsGraph constantWithScalar:3.0f
                                                           shape:@[ @1 ]
                                                        dataType:getMPSDataType(grad_output)];

      MPSGraphTensor* negativeThreeTensor = [mpsGraph constantWithScalar:-3.0f
                                                                   shape:@[ @1 ]
                                                                dataType:getMPSDataType(grad_output)];

      MPSGraphTensor* halfTensor = [mpsGraph constantWithScalar:0.5f
                                                          shape:@[ @1 ]
                                                       dataType:getMPSDataType(grad_output)];

      MPSGraphTensor* tempTensor = [mpsGraph divisionWithPrimaryTensor:inputTensor
                                                       secondaryTensor:threeTensor
                                                                  name:nil];

      MPSGraphTensor* weightedTensor = [mpsGraph additionWithPrimaryTensor:tempTensor
                                                           secondaryTensor:halfTensor
                                                                      name:nil];

      MPSGraphTensor* lessThanMinPredicateTensor = [mpsGraph lessThanOrEqualToWithPrimaryTensor:inputTensor
                                                                                secondaryTensor:negativeThreeTensor
                                                                                           name:nil];

      MPSGraphTensor* lessThanMaxPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                       secondaryTensor:threeTensor
                                                                                  name:nil];

      MPSGraphTensor* lessThanMaxGradTensor = [mpsGraph selectWithPredicateTensor:lessThanMaxPredicateTensor
                                                              truePredicateTensor:weightedTensor
                                                             falsePredicateTensor:unitTensor
                                                                             name:nil];

      MPSGraphTensor* gradTensor = [mpsGraph selectWithPredicateTensor:lessThanMinPredicateTensor
                                                   truePredicateTensor:zeroTensor
                                                  falsePredicateTensor:lessThanMaxGradTensor
                                                                  name:nil];
      MPSGraphTensor* gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradTensor
                                                                  secondaryTensor:gradOutputTensor
                                                                             name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, selfPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, gradInputPlaceholder);
  }
  return grad_input;
}
} // namespace at::native
