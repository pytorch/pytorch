//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/TensorUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/ConvUtils.h>
#include <torch/library.h>

namespace at::native {

void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                    NSUInteger strideInX, NSUInteger strideInY,
                    NSUInteger dilationRateInX, NSUInteger dilationRateInY,
                    NSUInteger paddingHorizontal, NSUInteger paddingVertical,
                    c10::MemoryFormat memory_format, NSUInteger groups) {
  descriptor_.strides = @[@1, [[NSNumber alloc] initWithInteger: strideInY],
                              [[NSNumber alloc] initWithInteger: strideInX]];
  descriptor_.dilationRates = @[@1, [[NSNumber alloc] initWithInteger: dilationRateInY],
                                      [[NSNumber alloc] initWithInteger: dilationRateInX]];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[@0, @0, [[NSNumber alloc] initWithInteger: paddingVertical], [[NSNumber alloc]
                                                            initWithInteger: paddingVertical], [[NSNumber alloc]
                                                            initWithInteger: paddingHorizontal], [[NSNumber alloc]
                                                            initWithInteger: paddingHorizontal]];
  descriptor_.channelDimensionIndex = -3LL;
}

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
    if(bias_defined) {
      bias_shape_key = to_string(bias_shape[0]);
    } else {
      bias_shape_key = "nobias";
    }

    string key = "mps_convolution:" + to_string(stride[0]) + ":" + to_string(stride[1]) + ":"
                                    + to_string(dilation[0]) + ":" + to_string(dilation[1]) + ":"
                                    + to_string(padding[0]) + ":" + to_string(padding[1]) + ":"
                                    + to_string(groups) + ":" +  mem_format_key
                                    + mps::getTensorsStringKey({input_t, weight_t}) + ":"
                                    + to_string(bias_defined) + ":" + bias_shape_key;
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    MPSShape* inputShape = mps::getMPSShape(input_t, memory_format);
    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphConvolution2DOpDescriptor *conv2dDescriptor_ =[[MPSGraphConvolution2DOpDescriptor new] autorelease];
          MPSGraphDepthwiseConvolution3DOpDescriptor *depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
          MPSShape* weightShape = mps::getMPSShape(weight_t);
          bool isDepthwiseConv = ((groups > 1 && (weightShape[1].intValue == 1)) &&
                                  inputShape.count >= 4 && weightShape.count >= 4  && !is_channels_last);
          if(isDepthwiseConv) {
            fill_depthwise_conv_desc(depthWiseConv3dDescriptor_, stride[1], stride[0],
                                            dilation[1], dilation[0],
                                            padding[1], padding[0],
                                            memory_format, groups);
          } else {
            fill_conv_desc(conv2dDescriptor_, stride[1], stride[0],
                                      dilation[1], dilation[0],
                                      padding[1], padding[0],
                                      memory_format, groups);
          }

          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(input_t.scalar_type()), inputShape);
          MPSGraphTensor* weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

          MPSGraphTensor* biasTensor = nil;
          if(bias_defined) {
            biasTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType((bias_opt.value()).scalar_type()));
          }

          MPSGraphTensor* outputTensor;
          if(isDepthwiseConv) {
              MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor dimension:-3 withDimension:-4 name:nil];
              outputTensor = [mpsGraph depthwiseConvolution3DWithSourceTensor: inputTensor
                                                                weightsTensor: weightTransposeTensor
                                                                   descriptor: depthWiseConv3dDescriptor_
                                                                         name: nil];
          } else {
              outputTensor = [mpsGraph convolution2DWithSourceTensor: inputTensor
                                                                 weightsTensor: weightTensor
                                                                    descriptor: conv2dDescriptor_
                                                                          name: nil];
          }

          if (is_channels_last) {
            outputTensor = mps::convertNHWCtoNCHW(mpsGraph, outputTensor);
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
    IntArrayRef input_size, const Tensor& grad_output_, const Tensor& weight_,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  namespace native_mps = at::native::mps;
  using namespace mps;
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{ grad_output_, "grad_output", 1 },
            weight{ weight_, "weight", 2 };
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  auto memory_format = grad_output_.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast);
  Tensor grad_output_t = grad_output_.contiguous(memory_format);
  Tensor weight_t = weight_.contiguous(memory_format);
  MPSShape* weightShape = getMPSShape(weight_);
  auto grad_input_t = at::empty( input_size, grad_output_t.options(), c10::nullopt);

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

    MPSShape* gradOutputShape = getMPSShape(grad_output_t, memory_format);
    MPSShape* mps_input_shape = getMPSShape(input_size);
    NSString* ns_shape_key = [[gradOutputShape valueForKey:@"description"] componentsJoinedByString:@","];
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

          MPSGraphConvolution2DOpDescriptor *conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
          MPSGraphDepthwiseConvolution3DOpDescriptor *depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new]  autorelease];

          MPSShape* weightOutputShape = mps::getMPSShape(weight_t);
          // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
          bool isDepthwiseConv = ((groups > 1 && (weightOutputShape[1].intValue == 1)) &&
                                  gradOutputShape.count >= 4 && weightOutputShape.count >= 4 && !is_channels_last);

          if(isDepthwiseConv) {
            fill_depthwise_conv_desc(depthWiseConv3dDescriptor_, stride[1], stride[0],
                                              dilation[1], dilation[0],
                                              padding[1], padding[0],
                                              at::MemoryFormat::Contiguous, groups);
          } else {
            fill_conv_desc(conv2dDescriptor_, stride[1], stride[0],
                                        dilation[1], dilation[0],
                                        padding[1], padding[0],
                                        at::MemoryFormat::Contiguous, groups);
          }

          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(grad_output_t.scalar_type()), gradOutputShape);
          MPSGraphTensor* weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(weight_t.scalar_type()), weightShape);

          MPSGraphTensor *gradOutputTensorTranspose = gradOutputTensor;
          if (is_channels_last && grad_output_t.is_contiguous() && !grad_output_t.is_view()) {
            gradOutputTensorTranspose = mps::convertNHWCtoNCHW(mpsGraph, gradOutputTensorTranspose);
          }
          MPSGraphTensor* gradInputTensor;
          if(isDepthwiseConv) {
              MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor dimension:-3 withDimension:-4 name:nil];
              gradInputTensor = [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                                weightsTensor:weightTransposeTensor
                                                                                  outputShape:mps_input_shape
                                                                                   descriptor:depthWiseConv3dDescriptor_
                                                                                         name:nil];
          } else {
              gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                                weightsTensor:weightTensor
                                                                                  outputShape:mps_input_shape
                                                                 forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                         name:nil];
          }

          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t, gradOutputShape);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t, weightShape);
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
    IntArrayRef weight_size, const Tensor& grad_output_, const Tensor& input_,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  namespace native_mps = at::native::mps;
  using namespace mps;
  CheckedFrom c = "mps_convolution_backward_weights";
  auto memory_format = input_.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast);

  auto grad_output_t = grad_output_.to(memory_format);
  auto input_t = input_.to(memory_format);

  MPSShape* gradOutputShape = mps::getMPSShape(grad_output_t, memory_format);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };
  TensorArg input{ input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(
                          weight_size,
                          grad_output_t.scalar_type(),
                          c10::nullopt,
                          kMPS,
                          c10::nullopt,
                          c10::nullopt);
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
    NSString* ns_shape_key = [[gradOutputShape valueForKey:@"description"] componentsJoinedByString:@","];
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

          MPSGraphConvolution2DOpDescriptor *conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
          MPSGraphDepthwiseConvolution3DOpDescriptor *depthWiseConv3dDescriptor_ = [[MPSGraphDepthwiseConvolution3DOpDescriptor new]  autorelease];
          MPSShape* inputShape = mps::getMPSShape(input_t);
          bool isDepthwiseConv = ((groups > 1 && (mps_weight_shape[1].intValue == 1)) && inputShape.count >= 4 && mps_weight_shape.count >= 4 && !is_channels_last);

          if(isDepthwiseConv) {
            fill_depthwise_conv_desc(depthWiseConv3dDescriptor_, stride[1], stride[0],
                                                dilation[1], dilation[0],
                                                padding[1], padding[0],
                                                at::MemoryFormat::Contiguous, groups);
          } else {
              fill_conv_desc(conv2dDescriptor_, stride[1], stride[0],
                                                  dilation[1], dilation[0],
                                                  padding[1], padding[0],
                                                  at::MemoryFormat::Contiguous, groups);
          }

          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSScalarType(grad_output_t.scalar_type()), gradOutputShape);
          MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_t);

          MPSGraphTensor *gradOutputTensorTranspose = gradOutputTensor;
          if (is_channels_last && grad_output_t.is_contiguous() && !grad_output_t.is_view()) {
            gradOutputTensorTranspose = mps::convertNHWCtoNCHW(mpsGraph, gradOutputTensorTranspose);
          }

          MPSGraphTensor* gradWeightTensor;
          if(isDepthwiseConv) {
              NSNumber* outputFeatChannelDim = mps_weight_shape[0];
              MPSShape* weightShapeTranspose = @[@1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3]];
              MPSGraphTensor* gradWeightTensorTranspose = [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                                              sourceTensor:inputTensor
                                                                                               outputShape:weightShapeTranspose
                                                                                                descriptor:depthWiseConv3dDescriptor_
                                                                                                      name:nil];
              gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
          } else {
              gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                                               sourceTensor:inputTensor
                                                                                                outputShape:mps_weight_shape
                                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                                       name:nil];
          }
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->gradWeightTensor_ = gradWeightTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t, gradOutputShape);
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
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
  TORCH_CHECK(input_t.dim() < 5, "ConvTranspose 3D is not supported on MPS");

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


} // namespace at::native
