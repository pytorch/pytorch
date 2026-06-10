//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/zeros.h>

namespace at::native {

using namespace mps;

// MPSNDArrayMatrixMultiplication and MPSGraph matrixMultiplication produce
// non-deterministic results for >2D fp16/bf16 inputs on Apple M5+ (Apple10 GPU family).
// Flatten to 2D to work around the issue (See https://github.com/pytorch/pytorch/issues/180776 )
static bool needs_nd_workaround(const Tensor& input) {
  static const bool is_m5_or_newer = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
  return input.dim() > 2 && is_m5_or_newer && (input.scalar_type() == kHalf || input.scalar_type() == kBFloat16);
}

static void _mps_linear_nograph(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
  bool is_bias_defined = bias.defined();

  auto mpsStream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();

  const std::string key = "mps_linear" + getTensorsStringKey({input, weight, bias}, true, true);
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->endKernelCoalescing();

      auto computeEncoder = mpsStream->commandEncoder();
      auto commandBuffer = mpsStream->commandBuffer();

      const auto mpsDataType = getMPSDataType(weight.scalar_type());

      auto inputNDArray = getMPSNDArray(input, input.sizes(), input.strides());
      auto outNDArray = getMPSNDArray(output, output.sizes(), output.strides());

      auto weightBuf = getMTLBufferStorage(weight);
      auto weightDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDataType shape:getMPSShape(weight.sizes())];
      weightDesc.preferPackedRows = YES;
      [weightDesc transposeDimension:0 withDimension:1];
      auto weightNDArray = [[[MPSNDArray alloc] initWithBuffer:weightBuf
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

  // Apple7/8 (M1/M2) MPSGraph silently produces wrong results when K exceeds 2^15;
  // M3+ are fine. Route through at::mm instead. See pytorch/pytorch#177116.
  constexpr int64_t mpsgraph_k_overflow = 32768;
  static const bool needs_k_overflow_fallback = !is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS);
  if (needs_k_overflow_fallback && input.size(-1) > mpsgraph_k_overflow && !input.is_complex() &&
      !weight.is_complex() && (!is_bias_defined || !bias.is_complex())) {
    const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;
    auto output_2d = at::mm(input_2d, weight.t());
    if (is_bias_defined) {
      output_2d.add_(bias);
    }
    auto reshaped = output_2d.view(output_size);
    return weight_arg.dim() != 1 ? reshaped : reshaped.squeeze(-1);
  }

  const bool is_complex = input.is_complex() || weight.is_complex() || (is_bias_defined && bias.is_complex());

  // No-graph execution causes nonsense if these are non-contiguous.
  const bool is_contiguous = input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous();

  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS) && is_contiguous && !is_complex) {
    if (needs_nd_workaround(input) && (!is_bias_defined || bias.dim() <= 1)) {
      auto input2d = input.flatten(0, -2);
      auto output2d = output.flatten(0, -2);
      _mps_linear_nograph(input2d, weight, bias, output2d);
    } else {
      _mps_linear_nograph(input, weight, bias, output);
    }
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
      // Non-deterministic results for >2D fp16/bf16 on Apple10+
      if (!doReshape) {
        doReshape = needs_nd_workaround(input);
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

  if (grad_output.numel() == 0) {
    return at::zeros(input_size, grad_output.options());
  }

  const auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
  const auto grad_output_2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  return at::mm(grad_output_2d, weight_contig).view(input_size);
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  const auto grad_output_2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;

  if (grad_output.numel() == 0) {
    auto grad_weight = at::zeros({grad_output_2d.size(1), input_2d.size(1)}, grad_output.options());
    auto grad_bias = bias_defined ? at::zeros({grad_output_2d.size(1)}, grad_output.options()) : Tensor();
    return {grad_weight, grad_bias};
  }

  // Route through at::mm so the dispatcher can pick the Metal fallback for K-dim
  // overflow on Apple7/8 (M1/M2). See pytorch/pytorch#177116.
  auto grad_weight = at::mm(grad_output_2d.t(), input_2d.contiguous());
  // autocast promotes sum() to float32, but linear_backward's meta keeps grad_output's
  // dtype; cast back so inductor's baked-in dtype matches the runtime buffer.
  auto grad_bias = bias_defined ? grad_output_2d.sum(0).to(grad_output.scalar_type()) : Tensor();
  return {grad_weight, grad_bias};
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
