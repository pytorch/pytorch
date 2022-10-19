//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/TensorUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/ConvUtils.h>
#include <torch/library.h>

namespace at {
namespace native {

// Create convolution descriptor
void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                    NSUInteger strideInX, NSUInteger strideInY,
                    NSUInteger dilationRateInX, NSUInteger dilationRateInY,
                    NSUInteger paddingHorizontal, NSUInteger paddingVertical,
                    c10::MemoryFormat memory_format, NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ?
        MPSGraphTensorNamedDataLayoutNCHW : MPSGraphTensorNamedDataLayoutNHWC;

  // PyTorch always uses OIHW memory layout for weights
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

static
MPSShape* get_mps_conv_shape(const Tensor& tensor, bool is_channels_last) {
  if (is_channels_last) {
    const auto tensorSizes = tensor.sizes();
    const NSUInteger N = tensorSizes[0];
    const NSUInteger C = tensorSizes[1];
    const NSUInteger H = tensorSizes[2];
    const NSUInteger W = tensorSizes[3];
    return @[@(N), @(H), @(W), @(C)];
  }
  return at::native::mps::getMPSShape(tensor);
}

Tensor _mps_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(input_t.dim() < 5, "Conv3D is not supported on MPS");

  namespace native_mps = at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  bool bias_defined;

  if(bias_opt == c10::nullopt)
    bias_defined = false;
  else
   bias_defined = bias_opt->defined();

  auto memory_format = input_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast);
  auto output_t = at::empty(
                    conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation),
                    input->scalar_type(),
                    c10::nullopt,
                    kMPS,
                    c10::nullopt,
                    c10::nullopt);

  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{ output_t, "result", 0 };

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {

    IntArrayRef bias_shape;
    if(bias_defined)
      bias_shape = bias_opt.value().sizes();

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    string bias_shape_key;
    if(bias_defined)
      bias_shape_key = to_string(bias_shape[0]);
    else
      bias_shape_key = "nobias";

    string key = "mps_convolution:" + to_string(stride[0]) + ":" + to_string(stride[1]) + ":"
                                    + to_string(dilation[0]) + ":" + to_string(dilation[1]) + ":"
                                    + to_string(padding[0]) + ":" + to_string(padding[1]) + ":"
                                    + to_string(groups) + ":" +  mem_format_key
                                    + mps::getTensorsStringKey({input_t, weight_t}) + ":"
                                    + to_string(bias_defined) + ":" + bias_shape_key;
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    MPSShape* inputShape = get_mps_conv_shape(input_t, is_channels_last);
    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphConvolution2DOpDescriptor *descriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
          fill_conv_desc(descriptor_, stride[1], stride[0],
                                      dilation[1], dilation[0],
                                      padding[1], padding[0],
                                      memory_format, groups);

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(input_t.scalar_type()), inputShape);
          MPSGraphTensor* weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

          MPSGraphTensor* biasTensor = nil;
          if(bias_defined)
            biasTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType((bias_opt.value()).scalar_type()));

          MPSGraphTensor* outputTensor = [mpsGraph convolution2DWithSourceTensor: inputTensor
                                                                   weightsTensor: weightTensor
                                                                      descriptor: descriptor_
                                                                            name: nil];
          if (is_channels_last) {
            // NHWC -> NCHW
            outputTensor = [mpsGraph transposeTensor: [mpsGraph transposeTensor:outputTensor dimension:-1 withDimension:-2 name:nil]
                                           dimension: -2
                                       withDimension: -3
                                                name: nil];
          }

          if(bias_defined) {
            outputTensor = [mpsGraph additionWithPrimaryTensor: outputTensor
                                               secondaryTensor: biasTensor
                                                          name: nil];
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

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input_t, inputShape);
    auto weightsPlaceholder = native_mps::Placeholder(cachedGraph->weightTensor_, weight_t);
    auto biasPlaceholder = native_mps::Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d
    if(bias_defined)
      biasPlaceholder = native_mps::Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1}));
    auto outputPlaceholder = native_mps::Placeholder(cachedGraph->outputTensor_, *output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] initWithCapacity: 3] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    if(bias_defined) {
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return *output;
}

Tensor mps_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  namespace native_mps = at::native::mps;
  using namespace mps;
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  auto memory_format = grad_output_t.suggest_memory_format();
  auto grad_input_t = at::empty(
                    input_size,
                    grad_output->scalar_type(),
                    c10::nullopt,
                    kMPS,
                    c10::nullopt,
                    c10::nullopt);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  // Add backward with input
  @autoreleasepool {

    MPSStream* stream = getCurrentMPSStream();

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    MPSShape* mps_input_shape = getMPSShape(input_size);
    NSString* ns_shape_key = [[mps_input_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = "mps_convolution_backward_input:" + to_string(stride[0]) + ":" + to_string(stride[1]) + ":"
                                                   + to_string(dilation[0]) + ":" + to_string(dilation[1]) + ":"
                                                   + to_string(padding[0]) + ":" + to_string(padding[1]) + ":"
                                                   + to_string(groups) + ":" +  mem_format_key
                                                   + getTensorsStringKey({grad_output_t, weight_t}) + ":"
                                                   + string([ns_shape_key UTF8String]);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph* newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphConvolution2DOpDescriptor *descriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
          fill_conv_desc(descriptor_, stride[1], stride[0],
                                      dilation[1], dilation[0],
                                      padding[1], padding[0],
                                      memory_format, groups);

          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output_t);
          MPSGraphTensor* weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

          MPSGraphTensor* gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensor
                                                                                            weightsTensor:weightTensor
                                                                                              outputShape:mps_input_shape
                                                                             forwardConvolutionDescriptor:descriptor_
                                                                                                     name:nil];

          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, *grad_input);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      weightsPlaceholder.getMPSGraphTensor() : weightsPlaceholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
  return *grad_input;
}

Tensor mps_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output_t, const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  namespace native_mps = at::native::mps;
  using namespace mps;
  CheckedFrom c = "mps_convolution_backward_weights";
  auto memory_format = input_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast);
  MPSShape* inputShape = get_mps_conv_shape(input_t, is_channels_last);
  MPSShape* gradOutputShape = get_mps_conv_shape(grad_output_t, is_channels_last);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };
  TensorArg input{ input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(weight_size, grad_output_t.options(), c10::nullopt);
  TensorArg grad_weight{ grad_weight_t, "result", 0 };

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  @autoreleasepool {

    MPSStream* stream = getCurrentMPSStream();

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    MPSShape* mps_weight_shape = getMPSShape(weight_size);
    NSString* ns_shape_key = [[mps_weight_shape valueForKey:@"description"] componentsJoinedByString:@","];
    string key = "mps_convolution_backward_weights:" + to_string(stride[0]) + ":" + to_string(stride[1]) + ":"
                                                     + to_string(dilation[0]) + ":" + to_string(dilation[1]) + ":"
                                                     + to_string(padding[0]) + ":" + to_string(padding[1]) + ":"
                                                     + to_string(groups) + ":" +  mem_format_key
                                                     + getTensorsStringKey({grad_output_t, input_t}) + ":"
                                                     + string([ns_shape_key UTF8String]);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph* newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphConvolution2DOpDescriptor *descriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
          fill_conv_desc(descriptor_, stride[1], stride[0],
                                      dilation[1], dilation[0],
                                      padding[1], padding[0],
                                      memory_format, groups);

          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(grad_output_t.scalar_type()), gradOutputShape);
          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(input_t.scalar_type()), inputShape);

          MPSGraphTensor* gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensor
                                                                                                 sourceTensor:inputTensor
                                                                                                  outputShape:mps_weight_shape
                                                                                 forwardConvolutionDescriptor:descriptor_
                                                                                                         name:nil];

          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gradWeightTensor_ = gradWeightTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t, gradOutputShape);
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t, inputShape);
    auto outputPlaceholder = Placeholder(cachedGraph->gradWeightTensor_, grad_weight_t);

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return grad_weight_t;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mps_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = mps_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    }
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor mps_convolution_transpose_forward(
    const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  auto input_size = conv_input_size(grad_output.sizes(), weight.sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return mps_convolution_backward_input(input_size, grad_output, weight,
                                    padding, stride, dilation, groups, false);
}

Tensor _mps_convolution_transpose(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {

  auto output_t = mps_convolution_transpose_forward(
    input_t, weight_t, padding, output_padding, stride, dilation, groups);
  return output_t;

}

Tensor mps_convolution_transpose_backward_input(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups)
{
  return at::_mps_convolution(
    grad_output_t, weight_t, c10::nullopt, padding, stride, dilation, groups);
}

Tensor mps_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t,
      padding, stride, dilation, groups, false);
}


std::tuple<Tensor,Tensor> mps_convolution_transpose_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input = mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups);
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
}


} // namespace native
} // namespace at
