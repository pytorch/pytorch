//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>

namespace at::native {

using namespace mps;

Tensor _mps_linear(const Tensor& input, const Tensor& weight_arg, const c10::optional<Tensor>& bias_opt) {
  // wT = transpose(weight);
  // y=x*wT+b

  auto weight = (weight_arg.dim() == 1) ? weight_arg.view({1, weight_arg.size(0)}) : weight_arg;

  TORCH_CHECK(input.scalar_type() == ScalarType::Float || input.scalar_type() == ScalarType::Half,
              "MPS device does not support linear for non-float inputs");

  const Tensor& bias = *(at::borrow_from_optional_tensor(bias_opt));
  bool is_bias_defined = bias.defined();

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
      at::empty(output_size, input.scalar_type(), c10::nullopt, kMPS, c10::nullopt, input.suggest_memory_format());

  if (output.numel() == 0) {
    return output;
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
    string key = "mps_linear" + getTensorsStringKey({input, weight, bias});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);

      MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                              dimension:-1
                                                          withDimension:-2
                                                                   name:nil];
      MPSGraphTensor* outputTensor = nil;

      if (!is_bias_defined) {
        outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                       secondaryTensor:weightTransposeTensor
                                                                  name:nil];
      } else {
        MPSGraphTensor* inputFlattened = inputTensor;
        bool doReshape = false;
        // workaround to improve the performance with 3D+ inputs
        if (input_size.size() > 2 && input_size[0] > 1 && input_size[1] >= 1 && input_size[1] <= 32 &&
            bias.dim() <= 1) {
          doReshape = true;
          inputFlattened = [mpsGraph flatten2DTensor:inputTensor axis:-1 name:nil];
        }

        newCachedGraph->biasTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, bias);
        MPSGraphTensor* xMulWTTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputFlattened
                                                                       secondaryTensor:weightTransposeTensor
                                                                                  name:nil];
        MPSGraphTensor* biasedTensor = [mpsGraph additionWithPrimaryTensor:xMulWTTensor
                                                           secondaryTensor:newCachedGraph->biasTensor_
                                                                      name:nil];
        outputTensor = doReshape ? [mpsGraph reshapeTensor:biasedTensor withShape:getMPSShape(output_size) name:nil]
                                 : biasedTensor;
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
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  // Shave off '1' present at the end of the shape
  if (weight_arg.dim() == 1) {
    // Number of elements in new output shape
    auto output_sizes = output.sizes();
    std::vector<int64_t> out_shape(output_sizes.begin(), output_sizes.end() - 1);
    return output.view(IntArrayRef(out_shape));
  }
  return output;
}

static Tensor _mps_linear_backward_input(IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(grad_output.is_mps(), "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && (weight.scalar_type() == kFloat || (weight.scalar_type() == kHalf)),
              "mps_linear_backward: unsupported weights data type: ",
              weight.scalar_type());

  TORCH_CHECK(grad_output.scalar_type() == ScalarType::Double || grad_output.scalar_type() == ScalarType::Float ||
                  grad_output.scalar_type() == ScalarType::Half,
              "MPS device does not support linear backward for non-float inputs");

  const Tensor weight_reshaped = weight.is_contiguous() ? weight : weight.contiguous();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  Tensor output = at::empty(
      input_size, grad_output.scalar_type(), c10::nullopt, kMPS, c10::nullopt, grad_output.suggest_memory_format());
  TORCH_CHECK(output.is_mps());
  if (grad_output.numel() == 0) {
    return output;
  }

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "mps_linear_backward_input" + getTensorsStringKey({grad_output, weight_reshaped});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_reshaped);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

      MPSGraphTensor* outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:gradOutputTensor
                                                                     secondaryTensor:weightTensor
                                                                                name:nil];

      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_reshaped);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weightPlaceholder.getMPSGraphTensor() : weightPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    return output;
  }
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(grad_output.scalar_type() == ScalarType::Float || grad_output.scalar_type() == ScalarType::Half,
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
                            c10::nullopt,
                            kMPS,
                            c10::nullopt,
                            grad_output.suggest_memory_format());
  Tensor bias = at::empty({grad_output_reshaped.size(1)},
                          grad_output.scalar_type(),
                          c10::nullopt,
                          kMPS,
                          c10::nullopt,
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
    string key = "mps_linear_backward_weights:" + to_string(bias_defined) + ":" +
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

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      weightPlaceholder.getMPSGraphTensor() : weightPlaceholder.getMPSGraphTensorData()
    };

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [NSMutableDictionary dictionary];
    results[outputPlaceholder.getMPSGraphTensor()] = outputPlaceholder.getMPSGraphTensorData();
    if (bias_defined)
      results[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();

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
