//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>

namespace at::native {

using namespace mps;

static void _mps_linear_nograph(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
  bool is_bias_defined = bias.defined();

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  const std::string key = "mps_linear" + getTensorsStringKey({input, weight, bias}, true, true);
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->endKernelCoalescing();

      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();

      MPSDataType mpsDataType = getMPSDataType(weight.scalar_type());

      auto inputNDArray = getMPSNDArray(input, input.sizes(), input.strides());
      auto outNDArray = getMPSNDArray(output, output.sizes(), output.strides());

      id<MTLBuffer> weightBuf = getMTLBufferStorage(weight);
      MPSNDArrayDescriptor* weightDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDataType
                                                                                shape:getMPSShape(weight.sizes())];
      weightDesc.preferPackedRows = YES;
      [weightDesc transposeDimension:0 withDimension:1];
      MPSNDArray* weightNDArray = [[[MPSNDArray alloc] initWithBuffer:weightBuf
                                                               offset:weight.storage_offset() * weight.element_size()
                                                           descriptor:weightDesc] autorelease];

      if (is_bias_defined) {
        auto biasNDArray = getMPSNDArray(bias, bias.sizes(), bias.strides());
        auto cachedKernel = LookUpOrCreateCachedKernel<MPSCachedKernel>(key, [&]() {
          return [[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:3] autorelease];
        });
        auto kernel = cachedKernel->kernel<MPSNDArrayMatrixMultiplication>();

        getMPSProfiler().beginProfileKernel(kernel, "mps_linear", {input, weight, bias});
        [kernel encodeToCommandEncoder:computeEncoder
                         commandBuffer:commandBuffer
                          sourceArrays:@[ inputNDArray, weightNDArray, biasNDArray ]
                      destinationArray:outNDArray];
        getMPSProfiler().endProfileKernel(kernel);
      } else {
        auto cachedKernel = LookUpOrCreateCachedKernel<MPSCachedKernel>(key, [&]() {
          return [[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2] autorelease];
        });
        auto kernel = cachedKernel->kernel<MPSNDArrayMatrixMultiplication>();
        getMPSProfiler().beginProfileKernel(kernel, "mps_linear", {input, weight, bias});
        [kernel encodeToCommandEncoder:computeEncoder
                         commandBuffer:commandBuffer
                          sourceArrays:@[ inputNDArray, weightNDArray ]
                      destinationArray:outNDArray];
        getMPSProfiler().endProfileKernel(kernel);
      }
    }
  });
}

Tensor _mps_linear(const Tensor& input, const Tensor& weight_arg, const std::optional<Tensor>& bias_opt) {
  // wT = transpose(weight);
  // y=x*wT+b

  TORCH_CHECK(supportedFloatingOrComplexType(input), "MPS device does not support linear for non-float inputs");
  TORCH_CHECK(input.is_mps(), "Tensor for argument input is on ", input.device(), " but expected on mps");
  TORCH_CHECK(supportedFloatingOrComplexType(weight_arg), "MPS device does not support linear for non-float weights");
  TORCH_CHECK(weight_arg.is_mps(), "Tensor for argument weight is on ", weight_arg.device(), " but expected on mps");
  TORCH_CHECK((input.scalar_type() != kComplexFloat && input.scalar_type() != kComplexHalf),
              "mps linear does not support complex types");

  const Tensor& bias = *(at::borrow_from_optional_tensor(bias_opt));
  const bool is_bias_defined = bias.defined();
  if (is_bias_defined) {
    TORCH_CHECK(bias.is_mps(), "Tensor for argument bias is on ", bias.device(), " but expected on mps");
    TORCH_CHECK(supportedFloatingOrComplexType(bias), "MPS device does not support linear for non-float bias");
  }

  auto weight = (weight_arg.dim() == 1) ? weight_arg.unsqueeze(0) : weight_arg;

  auto input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  TORCH_CHECK(input.size(-1) == weight_arg.size(-1),
              "linear(): input and weight.T shapes cannot be multiplied (",
              input.size(-2),
              "x",
              input.size(-1),
              " and ",
              weight_arg.size(-1),
              "x",
              weight_arg.size(-2),
              ")");

  if (is_bias_defined) {
    // Check bias and output shapes compatibility only.
    inferExpandGeometry_dimvector(bias.sizes(), bias.strides(), output_size);
  }

  Tensor output =
      at::empty(output_size, input.scalar_type(), std::nullopt, kMPS, std::nullopt, input.suggest_memory_format());

  if (output.numel() == 0) {
    return output;
  }

  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS)) {
    _mps_linear_nograph(input, weight, bias, output);
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }
  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    std::string key = "mps_linear" + getTensorsStringKey({input, weight, bias});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);

      MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                              dimension:-1
                                                          withDimension:-2
                                                                   name:nil];
      // matrixMultiplicationWithPrimary crashes for 5D tensors, see https://github.com/pytorch/pytorch/issues/114942
      bool doReshape = input.dim() > 4;
      if (!doReshape && is_bias_defined) {
        // workaround to improve the performance with 3D+ inputs
        doReshape =
            input_size.size() > 2 && input_size[0] > 1 && input_size[1] >= 1 && input_size[1] <= 32 && bias.dim() <= 1;
      }
      auto inputFlattened = doReshape ? [mpsGraph flatten2DTensor:inputTensor axis:-1 name:nil] : inputTensor;
      auto outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputFlattened
                                                          secondaryTensor:weightTransposeTensor
                                                                     name:nil];

      if (is_bias_defined) {
        newCachedGraph->biasTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, bias);
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor
                                           secondaryTensor:newCachedGraph->biasTensor_
                                                      name:nil];
      }
      if (doReshape) {
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:getMPSShape(output_size) name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    Placeholder biasPlaceholder = Placeholder();
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if (is_bias_defined) {
      biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias);
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  // Squeeze last dim of 1D linear
  return weight_arg.dim() != 1 ? output : output.squeeze(-1);
}

static Tensor _mps_linear_backward_input(IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(grad_output.is_mps(), "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && supportedFloatingOrComplexType(weight),
              "mps_linear_backward: unsupported weights data type: ",
              weight.scalar_type());

  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  const Tensor weight_reshaped = weight.is_contiguous() ? weight : weight.contiguous();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  Tensor output = at::empty(
      input_size, grad_output.scalar_type(), std::nullopt, kMPS, std::nullopt, grad_output.suggest_memory_format());
  TORCH_CHECK(output.is_mps());
  if (grad_output.numel() == 0) {
    return output;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "mps_linear_backward_input" + getTensorsStringKey({grad_output, weight_reshaped});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      newCachedGraph->weightTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, weight_reshaped);
      newCachedGraph->gradOutputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      // MPS matrixMultiplication crashes for 5D+ tensors on 14.2.1 with `New volume should match old volume`
      // See https://github.com/pytorch/pytorch/issues/114942 for more details
      bool needReshape = grad_output.dim() > 4;
      auto gradOutputTensor = needReshape
          ? [mpsGraph flatten2DTensor:newCachedGraph->gradOutputTensor_ axis:-1 name:nil]
          : newCachedGraph->gradOutputTensor_;

      auto outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:gradOutputTensor
                                                          secondaryTensor:newCachedGraph->weightTensor_
                                                                     name:nil];
      if (needReshape) {
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:getMPSShape(output) name:nil];
      }

      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_reshaped);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(weightPlaceholder, gradOutputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);

    return output;
  }
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
  };

  auto grad_output_reshaped =
      grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() != 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  TORCH_CHECK(grad_output_reshaped.is_mps());
  TORCH_CHECK(input_reshaped.is_mps());

  Tensor output = at::empty({grad_output_reshaped.size(1), input_reshaped.size(1)},
                            grad_output.scalar_type(),
                            std::nullopt,
                            kMPS,
                            std::nullopt,
                            grad_output.suggest_memory_format());
  Tensor bias = at::empty({grad_output_reshaped.size(1)},
                          grad_output.scalar_type(),
                          std::nullopt,
                          kMPS,
                          std::nullopt,
                          grad_output.suggest_memory_format());
  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(bias.is_mps());

  if (grad_output.numel() == 0) {
    output.zero_();
    bias.zero_();
    return std::tuple<Tensor, Tensor>{output, bias};
  }
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "mps_linear_backward_weights:" + std::to_string(bias_defined) + ":" +
        getTensorsStringKey({input_reshaped, weight, grad_output_reshaped});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_reshaped);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output_reshaped);

      MPSGraphTensor* gradOutputTransposeTensor = [mpsGraph transposeTensor:gradOutputTensor
                                                                  dimension:-1
                                                              withDimension:-2
                                                                       name:nil];

      // grad_weight
      MPSGraphTensor* outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:gradOutputTransposeTensor
                                                                     secondaryTensor:inputTensor
                                                                                name:nil];
      MPSGraphTensor* biasTensor = nil;
      if (bias_defined) {
        // grad_bias
        biasTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor axis:0 name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
      newCachedGraph->biasTensor_ = biasTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_reshaped);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_reshaped);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder, weightPlaceholder);
    auto results = bias_defined ? dictionaryFromPlaceholders(outputPlaceholder, biasPlaceholder)
                                : dictionaryFromPlaceholders(outputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    return std::tuple<Tensor, Tensor>{output, bias};
  }
}

std::tuple<Tensor, Tensor, Tensor> mps_linear_backward(const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& weight,
                                                       std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = _mps_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = _mps_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace at::native
