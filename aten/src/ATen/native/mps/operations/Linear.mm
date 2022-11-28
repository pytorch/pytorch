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

Tensor _mps_linear(
  const Tensor& input,
  const Tensor& weight_arg,
  const c10::optional<Tensor>& bias_opt) {
  // wT = transpose(weight);
  // y=x*wT+b

  using namespace mps;

  auto weight = (weight_arg.dim() == 1) ? weight_arg.view({1, weight_arg.size(0)}) : weight_arg;

  TORCH_CHECK(input.scalar_type() == ScalarType::Double
              || input.scalar_type() == ScalarType::Float
              || input.scalar_type() == ScalarType::Half, "MPS device does not support linear for non-float inputs");

  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  auto input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));
  Tensor output = at::native::empty_mps(output_size,
                                        input.scalar_type(),
                                        c10::nullopt,
                                        kMPS,
                                        c10::nullopt,
                                        input.suggest_memory_format());

  TORCH_CHECK(output.is_mps());

  if(output.numel() == 0) {
    return output;
  }

  MPSStream *stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  bool is_bias_defined = bias->defined();

  @autoreleasepool {

    MPSShape* wt_shape = getMPSShape(weight);
    string wt_key = string([[[wt_shape valueForKey:@"description"] componentsJoinedByString:@","] UTF8String]);
    string bias_key = "nobias";
    if(is_bias_defined) {
      bias_key = "bias";
    }

    string key = "mps_linear" + getTensorsStringKey({input, weight}) + ":" + bias_key;


    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {

      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {

          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
          MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);
          MPSGraphTensor* biasTensor = nil;

          if(is_bias_defined) {
            biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType((*bias).scalar_type()));
          }

          MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                  dimension:-1
                                                              withDimension:-2
                                                                       name:nil];

          MPSGraphTensor* outputTensor = nil;

          if (!is_bias_defined)
          {
            outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                           secondaryTensor:weightTransposeTensor
                                                                      name:nil];
          }
          else
          {
            MPSGraphTensor* xMulWTTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                                           secondaryTensor:weightTransposeTensor
                                                                                      name:nil];
            outputTensor = [mpsGraph additionWithPrimaryTensor:xMulWTTensor
                                               secondaryTensor:biasTensor
                                                          name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->biasTensor_ = biasTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    Placeholder biasPlaceholder = Placeholder();
    if(is_bias_defined)
      biasPlaceholder = Placeholder(cachedGraph->biasTensor_, *bias);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =[NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()]   = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if (is_bias_defined)
        feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  // Shave off '1' present at the end of the shape
  if(weight_arg.dim() == 1) {
    // Number of elements in new output shape
    auto output_sizes = output.sizes();
    std::vector<int64_t> out_shape(output_sizes.begin(), output_sizes.end()-1);
    return output.view(IntArrayRef(out_shape));
  }
  else
    return output;
}

Tensor _mps_linear_backward_input(
    IntArrayRef    input_size,
    const Tensor & grad_output,
    const Tensor & weight)
{
  TORCH_CHECK(grad_output.is_mps(),
      "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && weight.scalar_type() == kFloat,
      "mps_linear_backward: weight needs to be a dense tensor");

  TORCH_CHECK(grad_output.scalar_type() == ScalarType::Double
              || grad_output.scalar_type() == ScalarType::Float
              || grad_output.scalar_type() == ScalarType::Half, "MPS device does not support linear backward for non-float inputs");

  const Tensor weight_reshaped = weight.is_contiguous() ? weight : weight.contiguous();

   struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *weightTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  Tensor output = at::native::empty_mps(input_size,
                                        grad_output.scalar_type(),
                                        c10::nullopt,
                                        kMPS,
                                        c10::nullopt,
                                        grad_output.suggest_memory_format());
  TORCH_CHECK(output.is_mps());

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  MPSStream *stream= getCurrentMPSStream();

  @autoreleasepool {

   string key = "mps_linear_backward_input" + mps::getTensorsStringKey({grad_output, weight_reshaped});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *weightTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, weight_reshaped);
          MPSGraphTensor *gradOutputTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output);

          MPSGraphTensor *outputTensor =
            [mpsGraph matrixMultiplicationWithPrimaryTensor: gradOutputTensor
                                           secondaryTensor: weightTensor
                                                      name: nil];

          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    mps::Placeholder weightPlaceholder = mps::Placeholder(cachedGraph->weightTensor_, weight_reshaped);
    mps::Placeholder gradOutputPlaceholder = mps::Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    mps::Placeholder outputPlaceholder = mps::Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      weightPlaceholder.getMPSGraphTensor() : weightPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    return output;
  }
}

std::tuple<Tensor, Tensor> _mps_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined)
{
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
      "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(grad_output.scalar_type() == ScalarType::Double
              || grad_output.scalar_type() == ScalarType::Float
              || grad_output.scalar_type() == ScalarType::Half, "MPS device does not support linear backward for non-float inputs");

   struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *weightTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
    MPSGraphTensor *biasTensor_ = nil;
  };

  auto grad_output_reshaped = grad_output.dim() != 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() != 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  TORCH_CHECK(grad_output_reshaped.is_mps());
  TORCH_CHECK(input_reshaped.is_mps());

  Tensor output = at::native::empty_mps({grad_output_reshaped.size(1), input_reshaped.size(1)},
                                        grad_output.scalar_type(),
                                        c10::nullopt,
                                        kMPS,
                                        c10::nullopt,
                                        grad_output.suggest_memory_format());
  Tensor bias = at::native::empty_mps({grad_output_reshaped.size(1)},
                                        grad_output.scalar_type(),
                                        c10::nullopt,
                                        kMPS,
                                        c10::nullopt,
                                        grad_output.suggest_memory_format());
  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(bias.is_mps());

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  MPSStream *stream= getCurrentMPSStream();

  @autoreleasepool {

   string key = "mps_linear_backward_weights:" + to_string(bias_defined) + ":" +
                                                 mps::getTensorsStringKey({input_reshaped, weight, grad_output_reshaped});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *inputTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, input_reshaped);
          MPSGraphTensor *weightTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, weight);
          MPSGraphTensor *gradOutputTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output_reshaped);

          MPSGraphTensor *gradOutputTransposeTensor =
            [mpsGraph transposeTensor: gradOutputTensor
                            dimension: -1
                        withDimension: -2
                                 name: nil];

          // grad_weight
          MPSGraphTensor *outputTensor =
            [mpsGraph matrixMultiplicationWithPrimaryTensor: gradOutputTransposeTensor
                                            secondaryTensor: inputTensor
                                                       name: nil];
          MPSGraphTensor *biasTensor = nil;
          if (bias_defined)
          {
              // grad_bias
              biasTensor = [mpsGraph reductionSumWithTensor: gradOutputTensor
                                                       axis: 0
                                                       name: nil];

          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
          newCachedGraph->biasTensor_ = biasTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    mps::Placeholder inputPlaceholder = mps::Placeholder(cachedGraph->inputTensor_, input_reshaped);
    mps::Placeholder weightPlaceholder = mps::Placeholder(cachedGraph->weightTensor_, weight);
    mps::Placeholder gradOutputPlaceholder = mps::Placeholder(cachedGraph->gradOutputTensor_, grad_output_reshaped);
    mps::Placeholder outputPlaceholder = mps::Placeholder(cachedGraph->outputTensor_, output);
    mps::Placeholder biasPlaceholder = mps::Placeholder(cachedGraph->biasTensor_, bias);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      weightPlaceholder.getMPSGraphTensor() : weightPlaceholder.getMPSGraphTensorData()
    };

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [NSMutableDictionary dictionary];
    results[outputPlaceholder.getMPSGraphTensor()] = outputPlaceholder.getMPSGraphTensorData();
    if (bias_defined)
      results[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    return std::tuple<Tensor, Tensor>{ output, bias };
  }
}


std::tuple<Tensor, Tensor, Tensor> mps_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = _mps_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = _mps_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at
