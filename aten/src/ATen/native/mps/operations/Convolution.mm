//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ConvUtils.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_mps_convolution_native.h>
#include <ATen/ops/_mps_convolution_transpose_native.h>
#include <ATen/ops/mps_convolution_backward_native.h>
#include <ATen/ops/mps_convolution_transpose_backward_native.h>

#if !defined(__MAC_13_2) && (!defined(MAC_OS_X_VERSION_13_2) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_2))

@implementation FakeMPSGraphConvolution3DOpDescriptor
- (nonnull id)copyWithZone:(nullable NSZone*)zone {
  return self;
}

@end

#endif

namespace at::native {

// Create 3D convolution descriptor
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;

  // TODO: Program the padding style
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;

  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;

  // PyTorch always uses NCDHW memory layout for 3D tensors
  descriptor_.dataLayout = (MPSGraphTensorNamedDataLayout)7L; // MPSGraphTensorNamedDataLayoutNCDHW;

  // PyTorch always uses OIDHW memory layout for 3D weights
  descriptor_.weightsLayout = (MPSGraphTensorNamedDataLayout)9L; // MPSGraphTensorNamedDataLayoutOIDHW;

  descriptor_.groups = groups; // not yet tested in Xcode/C++
}

static void fill_depthwise_conv_desc(MPSGraphDepthwiseConvolution3DOpDescriptor* descriptor_,
                                     NSUInteger strideInX,
                                     NSUInteger strideInY,
                                     NSUInteger dilationRateInX,
                                     NSUInteger dilationRateInY,
                                     NSUInteger paddingHorizontal,
                                     NSUInteger paddingVertical,
                                     c10::MemoryFormat memory_format,
                                     NSUInteger groups) {
  descriptor_.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:strideInY], [[NSNumber alloc] initWithInteger:strideInX] ];
  descriptor_.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:dilationRateInY], [[NSNumber alloc] initWithInteger:dilationRateInX] ];

  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingVertical],
    [[NSNumber alloc] initWithInteger:paddingHorizontal],
    [[NSNumber alloc] initWithInteger:paddingHorizontal]
  ];
  descriptor_.channelDimensionIndex = -3LL;
}

// Create convolution descriptor
static void fill_conv_desc(MPSGraphConvolution2DOpDescriptor* descriptor_,
                           NSUInteger strideInX,
                           NSUInteger strideInY,
                           NSUInteger dilationRateInX,
                           NSUInteger dilationRateInY,
                           NSUInteger paddingHorizontal,
                           NSUInteger paddingVertical,
                           c10::MemoryFormat memory_format,
                           NSUInteger groups) {
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

  descriptor_.dataLayout = (memory_format == at::MemoryFormat::Contiguous) ? MPSGraphTensorNamedDataLayoutNCHW
                                                                           : MPSGraphTensorNamedDataLayoutNHWC;

  // PyTorch always uses OIHW memory layout for weights
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
  descriptor_.groups = groups;
}

static MPSGraphTensor* permuteTensor(MPSGraph* graph, MPSGraphTensor* inputTensor, NSArray* permuteOrder) {
    NSUInteger srcRank = [[inputTensor shape] count];
    if (srcRank != [permuteOrder count]) {
        return nil;
    }
    MPSGraphTensor* outputTensor = inputTensor;
    outputTensor = [graph transposeTensor:outputTensor permutation:permuteOrder name:nil];
}

static MPSGraphTensor* reshapePermuteReshape(MPSGraph* mpsGraph, MPSGraphTensor* tensor__, MPSShape* reshape1, MPSShape* permutation, MPSShape* reshape2) {
    MPSGraphTensor *tensor_ = [mpsGraph reshapeTensor:tensor__ withShape:reshape1 name:nil];
    MPSGraphTensor *tensor;
    if (@available(macOS 13.0, *)) {
        tensor = [mpsGraph transposeTensor:tensor_ permutation:permutation name:nil];
    } else {// Fallback on earlier versions
        tensor = permuteTensor(mpsGraph, tensor_, permutation);
    }
    return [mpsGraph reshapeTensor:tensor withShape:reshape2 name:nil];
}

static MPSGraphTensor* unfoldConvolution2d(MPSGraph* mpsGraph, MPSGraphTensor* input1, int64_t stride, int64_t padding, int64_t dilation, int64_t k_D, MPSDataType datatype, MPSShape* outshape, bool not_transp){
    
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:stride dilationRateInX:1 dilationRateInY:dilation groups:1 paddingLeft:0 paddingRight:0 paddingTop:padding paddingBottom:padding paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor* ones_k = [mpsGraph constantWithScalar:1.0f shape:@[@(k_D),@(k_D)]  dataType:datatype];
        MPSGraphTensor* eye_k_ = [mpsGraph bandPartWithTensor:ones_k numLower:0 numUpper:0 name:nil];
    MPSGraphTensor* eye_k;
    if (not_transp) {
        eye_k = [mpsGraph reshapeTensor:eye_k_ withShape:@[@(k_D),@1,@(k_D),@1] name:nil];
    } else{ //transpose unfold kernel
        eye_k = [mpsGraph reshapeTensor:eye_k_ withShape:@[@1,@(k_D),@(k_D),@1] name:nil];
    }
    if(outshape==nil){
        return [mpsGraph convolution2DWithSourceTensor:input1 weightsTensor:eye_k descriptor:conv2dDescriptor_ name:Nil];
    }
    else{
        return [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:input1 weightsTensor:eye_k outputShape:outshape forwardConvolutionDescriptor:conv2dDescriptor_ name:Nil];
    }
}

/*   for 3D Convs a more efficient two-step approach that uses unfold+conv2d is employed
 == equivalent pytorch / python code ==
 @staticmethod
     def forward(ctx, x, weight, shapes):
         B,in_C,in_D,in_H,in_W = x.shape
         out_C,_,k_D,k_H,k_W = weight.shape
         p_D,p_H,p_W = shapes[0].tolist()#padding
         s_D,s_H,s_W = shapes[1].tolist()#stride
         d_D,d_H,d_W = shapes[2].tolist()#dilation
         out_D,out_H,out_W = shapes[3].tolist()#shape_out
         groups,_,_ = shapes[4].tolist()
         weight2d = weight.view(out_C,-1,k_H,k_W)
         unfold_weight = torch.eye(k_D,k_D).to(device).view(k_D,1,k_D,1)
         x2d = F.conv2d(x.view(-1,1,in_D,in_H*in_W),unfold_weight,padding=(p_D,0),stride=(s_D,1),dilation=(d_D,1))
         x2d_ = x2d.view(B,in_C,k_D,out_D,in_H,in_W).permute(0,3,1,2,4,5).reshape(B*out_D,in_C*k_D,in_H,in_W)
         out = F.conv2d(x2d_,weight2d,padding=(p_H,p_W),stride=(s_H,s_W),dilation=(d_H,d_W),groups=groups).view(B,out_D,out_C,out_H,out_W).permute(0,2,1,3,4)
         ctx.save_for_backward(x2d_,weight2d,unfold_weight,shapes)
         return out
 
 */
static Tensor _mps_convolution_impl(const Tensor& input_t_,
                                    const Tensor& weight_t,
                                    const std::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups,
                                    std::optional<IntArrayRef> input_shape) {
  const bool is_macOS_13_2_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS);
  const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  Tensor input_t = input_t_;
  if (!is_macOS_15_0_or_newer) {
    input_t = input_t.contiguous();
  }

  //TORCH_CHECK(((input_t.dim() < 5) || is_macOS_13_2_or_newer),
  //            "Conv3D is only supported on MPS for MacOS_13_2 or newer");
  bool is3DConv = input_t.dim() == 5;

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  bool bias_defined;

  if (bias_opt == std::nullopt)
    bias_defined = false;
  else
    bias_defined = bias_opt->defined();

  auto memory_format = input_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast) && !is3DConv;
  auto output_t =
      at::empty(input_shape.has_value() ? input_shape.value()
                                        : conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation),
                input->scalar_type(),
                std::nullopt,
                kMPS,
                std::nullopt,
                is_macOS_15_0_or_newer ? memory_format : MemoryFormat::Contiguous);
  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

  // TODO: MPS convolution kernel currently does not support output channels > 2^16
  for (auto elem : output_t.sizes()) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        elem <= (1 << 16),
        "Output channels > 65536 not supported at the MPS device. ",
        "As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` ",
        "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
        "on MPS.");
  }

  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    IntArrayRef bias_shape;
    if (bias_defined)
      bias_shape = bias_opt.value().sizes();

    string mem_format_key;
    switch (memory_format) {
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
    if (bias_defined) {
      bias_shape_key = std::to_string(bias_shape[0]);
    } else {
      bias_shape_key = "nobias";
    }

    string key;
    if (is3DConv) {
      key = "mps_3d_convolution:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(stride[2]) + ":" + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          mps::getTensorsStringKey({input_t, weight_t}) + ":" + std::to_string(bias_defined) + ":" + bias_shape_key;

    } else {
      key = "mps_convolution:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          mps::getTensorsStringKey({input_t, weight_t}) + ":" + std::to_string(bias_defined) + ":" + bias_shape_key;
    }

    MPSShape* inputShape = mps::getMPSShape(input_t, memory_format);
    MPSShape* outputShape = mps::getMPSShape(output_t, memory_format);
    MPSNDArray* inputNDArray = nil;
    MPSNDArray* outputNDArray = nil;

    if (input_t.is_contiguous(memory_format) && output_t.is_contiguous(memory_format) && is_macOS_15_0_or_newer) {
      inputNDArray = getMPSNDArray(input_t, inputShape);
      outputNDArray = getMPSNDArray(*output, outputShape);
    }

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* weightShape = mps::getMPSShape(weight_t);
      bool isDepthwiseConv = ((groups > 1 && (weightShape[1].intValue == 1)) && inputShape.count >= 4 &&
                              weightShape.count >= 4 && !is_channels_last);

      MPSGraphTensor* inputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t.scalar_type()), inputShape);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor;
      if (is3DConv) {
          std::vector<int64_t> output_shape = conv_output_size(input_t.sizes(), weight_t.sizes(), padding, stride, dilation);
          
          const int64_t B = inputShape[0].intValue;
          const int64_t in_C = inputShape[1].intValue;
          const int64_t out_C =  weightShape[0].intValue;
          
          const int64_t k_D = weightShape[2].intValue;
          const int64_t k_H = weightShape[3].intValue;
          const int64_t k_W = weightShape[4].intValue;
          
          const int64_t in_D = inputShape[2].intValue;
          const int64_t in_H = inputShape[3].intValue;
          const int64_t in_W = inputShape[4].intValue;
          
          const int64_t out_D = output_shape[2];
          const int64_t out_H = output_shape[3];
          const int64_t out_W = output_shape[4];
          
          MPSGraphTensor *weights1 = [mpsGraph reshapeTensor:weightTensor withShape:@[@(out_C),@(in_C*k_D/groups),@(k_H),@(k_W)] name:nil];
          MPSGraphTensor *unfold;
          //actual convolution is performed as conv2D with kernel in depth dimension as additional input channels
          //the spatial depth dimension will been moved to batch dimension (and needs to be permuted again afterwards)
          MPSGraphConvolution2DOpDescriptor* conv2dDescriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[2] strideInY:stride[1] dilationRateInX:dilation[2] dilationRateInY:dilation[1] groups:groups paddingLeft:padding[2] paddingRight:padding[2] paddingTop:padding[1] paddingBottom:padding[1] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
          if((k_D==1)&(k_H==1)&(k_W==1)&(stride[0]==1)&(stride[1]==1)&(stride[2]==1)&(padding[0]==0)&(padding[1]==0)&(padding[2]==0)){ //1x1x1 convolution (aka linear layer) no permutation/unfold necessary
              MPSGraphTensor *input1 = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B),@(in_C),@(in_D),@(in_H*in_W)] name:nil];
              MPSGraphTensor *output_ = [mpsGraph convolution2DWithSourceTensor:input1 weightsTensor:weights1 descriptor:conv2dDescriptor name:Nil];
              outputTensor = [mpsGraph reshapeTensor:output_ withShape:@[@(B),@(out_C),@(in_D),@(in_H),@(in_W)] name:nil];

          }
          else{
              
              if((k_D!=1)|(stride[0]!=1)|(padding[0]!=0)){
                  MPSGraphTensor *input1 = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B*in_C),@1,@(in_D),@(in_H*in_W)] name:nil];
                  MPSGraphTensor *unfold_ = unfoldConvolution2d(mpsGraph, input1, stride[0], padding[0], dilation[0], k_D, getMPSScalarType(input_t),nil,true); //2nd last arg nil = fwd, last arg true not transposed
                  unfold = reshapePermuteReshape(mpsGraph, unfold_, @[@(B),@(in_C),@(k_D),@(out_D),@(in_H),@(in_W)], @[@0,@3,@1,@2,@4,@5], @[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)]); //short-hand
              }
              else{ //special case for which no unfold from 3D to 2D is required and simple reshape/permute is equivalent
                  MPSGraphTensor *unfold_;
                  if (@available(macOS 13.0, *)) {
                      unfold_ = [mpsGraph transposeTensor:inputTensor permutation:@[@0,@2,@1,@3,@4] name:nil];
                  } else {// Fallback on earlier versions
                      unfold_ = permuteTensor(mpsGraph, inputTensor, @[@0,@2,@1,@3,@4]);
                  }
                  unfold = [mpsGraph reshapeTensor:unfold_ withShape:@[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)] name:nil];
              }
              
              MPSGraphTensor *output__ = [mpsGraph convolution2DWithSourceTensor:unfold weightsTensor:weights1 descriptor:conv2dDescriptor name:Nil];
              MPSGraphTensor *output_ = [mpsGraph reshapeTensor:output__ withShape:@[@(B),@(out_D),@(out_C),@(out_H),@(out_W)] name:nil];
              if (@available(macOS 13.0, *)) {
                  outputTensor = [mpsGraph transposeTensor:output_ permutation:@[@0,@2,@1,@3,@4] name:nil];
              } else {// Fallback on earlier versions
                  outputTensor = permuteTensor(mpsGraph, output_, @[@0,@2,@1,@3,@4]);
              }

          }

        /*outputTensor = [mpsGraph convolution3DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv3dDescriptor_
                                                          name:nil];*/
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 memory_format,
                                 groups);

        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        outputTensor = [mpsGraph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                          weightsTensor:weightTransposeTensor
                                                             descriptor:depthWiseConv3dDescriptor_
                                                                   name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       memory_format,
                       groups);

        outputTensor = [mpsGraph convolution2DWithSourceTensor:inputTensor
                                                 weightsTensor:weightTensor
                                                    descriptor:conv2dDescriptor_
                                                          name:nil];
      }

      MPSGraphTensor* biasTensor = nil;
      if (bias_defined) {
        biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(bias_opt.value()));
      }

      if (is_channels_last && !is_macOS_15_0_or_newer) {
        outputTensor = mps::convertNHWCtoNCHW(mpsGraph, outputTensor);
      }

      if (bias_defined) {
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor secondaryTensor:biasTensor name:nil];
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = inputNDArray ? Placeholder(cachedGraph->inputTensor_, inputNDArray)
                                         : Placeholder(cachedGraph->inputTensor_, input_t, inputShape);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
    auto biasPlaceholder = Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d or conv3d
    if (bias_defined) {
      if (is3DConv) {
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1, 1}));
      } else {
        if (is_channels_last && is_macOS_15_0_or_newer) {
          biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, 1, 1, bias_shape[0]}));
        } else {
          biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1}));
        }
      }
    }
    auto outputPlaceholder = outputNDArray ? Placeholder(cachedGraph->outputTensor_, outputNDArray)
                                           : Placeholder(cachedGraph->outputTensor_, *output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    if (bias_defined) {
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return *output;
}

//        return _mps_convTrans3d_impl(input_t, weight_t, padding, output_padding, stride, dilation, groups);



static Tensor _mps_convTrans3d_impl(const Tensor& input_t,
                                    const Tensor& weight_t,
                                    //const c10::optional<Tensor>& bias_opt,
                                    IntArrayRef padding,
                                    IntArrayRef output_padding,
                                    IntArrayRef stride,
                                    IntArrayRef dilation,
                                    int64_t groups) {
  const bool is_macOS_13_2_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS);

    
    // === TODO CHECK in_channel != out_channel AND CHECK stride != 2 ===
  
  //TORCH_CHECK((is_macOS_13_2_or_newer),
  //            "ConvTransp3D is only supported on MPS for MacOS_13_2 or newer");

  TORCH_CHECK(isFloatingType(input_t.scalar_type()), "Convolution is supported only for Floating types");

  using namespace at::native::mps;
  CheckedFrom c = "mps_convolution";
  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  //bool bias_defined;

  //if (bias_opt == c10::nullopt)
  bool bias_defined = false;
  //else
  //  bias_defined = bias_opt->defined();

  auto memory_format = input_t.suggest_memory_format();
  bool is_channels_last = false;
    auto bwd_input_size =
        conv_input_size(input->sizes(), weight->sizes(), padding, output_padding, stride, dilation, groups);
//conv_output_size(input->sizes(), weight->sizes(), padding, stride, dilation)
  auto output_t =
      at::empty(bwd_input_size,
                input->scalar_type(),
                c10::nullopt,
                kMPS,
                c10::nullopt,
                c10::nullopt);

  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};

    //doesn't seem to work for transpose?! swap in-output?!
  //convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  auto stream = at::mps::getCurrentMPSStream();

  @autoreleasepool {
    IntArrayRef bias_shape;
    //if (bias_defined)
    //  bias_shape = bias_opt.value().sizes();

    string mem_format_key;
    switch (memory_format) {
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
    //if (bias_defined) {
    //  bias_shape_key = to_string(bias_shape[0]);
    //} else {
      bias_shape_key = "nobias";
    //}

    string key;
      key = "mps_3d_convTrans:" + to_string(stride[0]) + ":" + to_string(stride[1]) + ":" + to_string(stride[2]) +
          ":" + to_string(dilation[0]) + ":" + to_string(dilation[1]) + ":" + to_string(dilation[2]) + ":" +
          to_string(padding[0]) + ":" + to_string(padding[1]) + ":" + to_string(padding[2]) + ":" + to_string(groups) +
          ":" + mem_format_key + mps::getTensorsStringKey({input_t, weight_t}) + ":" + to_string(bias_defined) + ":" +
          bias_shape_key;

   
    MPSShape* inputShape = mps::getMPSShape(input_t, memory_format);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* weightShape = mps::getMPSShape(weight_t);
      bool isDepthwiseConv = false;

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(input_t), inputShape);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);
      MPSGraphTensor* outputTensor;

          
        
        const int64_t B = inputShape[0].intValue;
        const int64_t in_C = inputShape[1].intValue;
        const int64_t out_C = weightShape[1].intValue;
        const int64_t out_D = output_t.size(2);
        const int64_t out_H = output_t.size(3);
        const int64_t out_W = output_t.size(4);
        
        const int64_t k_D = weightShape[2].intValue;
        const int64_t k_H = weightShape[3].intValue;
        const int64_t k_W = weightShape[4].intValue;
        
        const int64_t in_D = inputShape[2].intValue;
        const int64_t in_H = inputShape[3].intValue;
        const int64_t in_W = inputShape[4].intValue;

/*        MPSGraphTensor *weights1__ = [mpsGraph reshapeTensor:weightTensor withShape:@[@(in_C),@(out_C),@(k_D),@(k_H),@(k_W)] name:nil];
        
        MPSGraphTensor *weights1_ = [mpsGraph transposeTensor:weights1__ permutation:@[@2,@0,@1,@3,@4] name:nil];
        MPSGraphTensor *weights1 = [mpsGraph reshapeTensor:weights1_ withShape:@[@(-1),@(out_C),@(k_H),@(k_W)] name:nil];*/
        MPSGraphTensor *weights1  = reshapePermuteReshape(mpsGraph, weightTensor, @[@(in_C),@(out_C),@(k_D),@(k_H),@(k_W)], @[@2,@0,@1,@3,@4], @[@(-1),@(out_C),@(k_H),@(k_W)]); //short-hand

        

        MPSGraphTensor *input1 = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B*in_C),@1,@(in_D),@(in_H*in_W)] name:nil];
        MPSGraphTensor *unfold_ = unfoldConvolution2d(mpsGraph, input1, stride[0], padding[0], dilation[0], k_D, getMPSScalarType(input_t),@[@(B*in_C),@(k_D),@(out_D),@(in_H*in_W)],false); //2nd last arg nil = fwd, last arg false=transpose
        MPSGraphTensor *unfold = reshapePermuteReshape(mpsGraph, unfold_, @[@(B),@(in_C),@(k_D),@(out_D),@(in_H),@(in_W)], @[@0,@3,@2,@1,@4,@5], @[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)]); //short-hand

        
        /*MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:stride[0] dilationRateInX:1 dilationRateInY:dilation[0] groups:1 paddingLeft:0 paddingRight:0 paddingTop:padding[0] paddingBottom:padding[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor* ones_k = [mpsGraph constantWithScalar:1.0f shape:@[@(k_D),@(k_D)]  dataType:getMPSScalarType(input_t)];
        MPSGraphTensor* eye_k_ = [mpsGraph bandPartWithTensor:ones_k numLower:0 numUpper:0 name:nil];
        MPSGraphTensor* eye_k = [mpsGraph reshapeTensor:eye_k_ withShape:@[@1,@(k_D),@(k_D),@1] name:nil];
        
        MPSGraphTensor *unfold___ = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:input1 weightsTensor:eye_k outputShape:@[@(B*in_C),@(k_D),@(out_D),@(in_H*in_W)] forwardConvolutionDescriptor:conv2dDescriptor_ name:Nil];
        MPSGraphTensor *unfold__ = [mpsGraph reshapeTensor:unfold___ withShape:@[@(B),@(in_C),@(k_D),@(out_D),@(in_H),@(in_W)] name:nil];
        MPSGraphTensor *unfold_ = [mpsGraph transposeTensor:unfold__ permutation:@[@0,@3,@2,@1,@4,@5] name:nil];
        MPSGraphTensor *unfold = [mpsGraph reshapeTensor:unfold_ withShape:@[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)] name:nil];*/
        
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[2] strideInY:stride[1] dilationRateInX:dilation[2] dilationRateInY:dilation[1] groups:1 paddingLeft:padding[2] paddingRight:padding[2] paddingTop:padding[1] paddingBottom:padding[1] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *output1__ = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:unfold weightsTensor:weights1 outputShape:@[@(B*out_D),@(out_C),@(out_H),@(out_W)] forwardConvolutionDescriptor:conv2dDescriptor name:Nil];
        MPSGraphTensor *output1_ = [mpsGraph reshapeTensor:output1__ withShape:@[@(B),@(out_D),@(out_C),@(out_H),@(out_W)] name:nil];
        if (@available(macOS 13.0, *)) {
            outputTensor = [mpsGraph transposeTensor:output1_ permutation:@[@0,@2,@1,@3,@4] name:nil];
        } else {// Fallback on earlier versions
            outputTensor = permuteTensor(mpsGraph, output1_, @[@0,@2,@1,@3,@4]);
        }
        

      MPSGraphTensor* biasTensor = nil;
      //if (bias_defined) {
      //  biasTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSDataType(bias_opt.value()));
      //}

      //if (is_channels_last) {
      //  outputTensor = mps::convertNHWCtoNCHW(mpsGraph, outputTensor);
      //}

      //if (bias_defined) {
      //  outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor secondaryTensor:biasTensor name:nil];
      //}
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->biasTensor_ = biasTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t, inputShape);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
    auto biasPlaceholder = Placeholder();
    // Reshape the bias to be broadcastable with output of conv2d or conv3d
    /*if (bias_defined) {
      if (is3DConv) {
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1, 1}));
      } else {
        biasPlaceholder = Placeholder(cachedGraph->biasTensor_, (bias_opt.value()).view({1, bias_shape[0], 1, 1}));
      }
    }*/
    auto outputPlaceholder = Placeholder(cachedGraph->outputTensor_, *output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    //if (bias_defined) {
    //  feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
   // }

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return *output;
}


Tensor _mps_convolution(const Tensor& input_t,
                        const Tensor& weight_t,
                        const std::optional<Tensor>& bias_opt,
                        IntArrayRef padding,
                        IntArrayRef stride,
                        IntArrayRef dilation,
                        int64_t groups) {
  return _mps_convolution_impl(input_t, weight_t, bias_opt, padding, stride, dilation, groups, std::nullopt);
}

/*   for 3D Convs a more efficient two-step approach that uses unfold+conv2d is employed
 == equivalent pytorch / python code ==
 @staticmethod
     def backward(ctx, gradient):
         x2d_,weight2d,unfold_weight,shapes = ctx.saved_tensors
         B,in_C,in_D,in_H,in_W = x.shape
         out_C,_,k_D,k_H,k_W = weight.shape
         p_D,p_H,p_W = shapes[0].tolist()#padding
         s_D,s_H,s_W = shapes[1].tolist()#stride
         d_D,d_H,d_W = shapes[2].tolist()#dilation
         out_D,out_H,out_W = shapes[3].tolist()#shape_out
         groups,_,_ = shapes[4].tolist()

         outback = gradient.permute(0,2,1,3,4).reshape(B*out_D,out_C,out_H,out_W)
         x2d_grad_ = -jacobian(lambda x: (F.conv2d(x,weight2d,padding=(p_H,p_W),dilation=(d_H,d_W),stride=(s_H,s_W),groups=groups)-outback)\
                               .pow(2).mul(0.5).sum(),torch.zeros(B*out_D,in_C*k_D,in_H,in_W))
         x2d_grad = x2d_grad_.reshape(B,out_D,in_C,k_D,in_H,in_W).permute(0,2,3,1,4,5).reshape(B*in_C,k_D,out_D,in_H*in_W)
         x_grad_ = -jacobian(lambda x: (F.conv2d(x,unfold_weight,padding=(p_D,0),dilation=(d_D,1),stride=(s_D,1))-x2d_grad)\
                             .pow(2).mul(0.5).sum(),torch.zeros(B*in_C,1,in_D,in_H*in_W))
         x_grad = x_grad_.view(B,in_C,in_D,in_H,in_W)
         w_grad = -jacobian(lambda w: (F.conv2d(x2d_,w,padding=(p_H,p_W),dilation=(d_H,d_W),stride=(s_H,s_W),groups=groups)-outback).pow(2).mul(0.5).sum(), torch.zeros(out_C,in_C*k_D//groups,k_H,k_W)).view(out_C,in_C//groups,k_D,k_H,k_W)

         return x_grad,w_grad,None #shapes has no grad
 */
static Tensor mps_convolution_backward_input(IntArrayRef input_size,
                                             const Tensor& grad_output_t,
                                             const Tensor& weight_t,
                                             IntArrayRef padding,
                                             IntArrayRef stride,
                                             IntArrayRef dilation,
                                             int64_t groups,
                                             bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = grad_output_t.dim() == 5;

  // TODO: MPS convolution kernel currently does not support output channels > 2^16
  for (auto elem : grad_output_t.sizes()) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        elem <= (1 << 16),
        "Output channels > 65536 not supported at the MPS device. ",
        "As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` ",
        "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
        "on MPS.");
  }

  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_input";
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});
  auto memory_format = grad_output_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast) && !is3DConv;
  auto grad_input_t = at::empty(input_size, grad_output_t.options(), std::nullopt);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
  };

  // Add backward with input
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    string mem_format_key;
    switch (memory_format) {
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
    string key;
    if (is3DConv) {
      key = "mps_3d_convolution_backward_input:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          ":" + std::to_string(stride[2]) + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, weight_t}) + ":" + string([ns_shape_key UTF8String]);

    } else {
      key = "mps_convolution_backward_input:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, weight_t}) + ":" + string([ns_shape_key UTF8String]);
    }
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight_t);

      MPSGraphTensor* gradOutputTensorTranspose = gradOutputTensor;
      if (is_channels_last) {
        gradOutputTensorTranspose = mps::convertNHWCtoNCHW(mpsGraph, gradOutputTensorTranspose);
      }
      MPSGraphTensor* gradInputTensor;
      MPSShape* weightOutputShape = mps::getMPSShape(weight_t);
      // Depthwise conv is input feature channels = groups. So I in OIHW has to be 1.
      bool isDepthwiseConv = ((groups > 1 && (weightOutputShape[1].intValue == 1)) && gradOutputShape.count >= 4 &&
                              weightOutputShape.count >= 4 && !is_channels_last);

      if (is3DConv) {
          const int64_t in_D = mps_input_shape[2].intValue;
          const int64_t in_H = mps_input_shape[3].intValue;
          const int64_t in_W = mps_input_shape[4].intValue;
          const int64_t B = mps_input_shape[0].intValue;
          const int64_t out_C = gradOutputShape[1].intValue;
          const int64_t out_D = gradOutputShape[2].intValue;
          const int64_t out_H = gradOutputShape[3].intValue;
          const int64_t out_W = gradOutputShape[4].intValue;
          const int64_t in_C = mps_input_shape[1].intValue;
          const int64_t k_D = weightOutputShape[2].intValue;
          const int64_t k_H = weightOutputShape[3].intValue;
          const int64_t k_W = weightOutputShape[4].intValue;

          MPSGraphConvolution2DOpDescriptor* conv2dDescriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[2] strideInY:stride[1] dilationRateInX:dilation[2] dilationRateInY:dilation[1] groups:groups paddingLeft:padding[2] paddingRight:padding[2] paddingTop:padding[1] paddingBottom:padding[1] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

          if((k_D==1)&(k_H==1)&(k_W==1)&(stride[0]==1)&(stride[1]==1)&(stride[2]==1)&(padding[0]==0)&(padding[1]==0)&(padding[2]==0)){ //1x1x1 convolution (aka linear layer) no permutation/unfold necessary
              MPSGraphTensor *output_grad = [mpsGraph reshapeTensor:gradOutputTensor withShape:@[@(B),@(out_C),@(out_D),@(out_H*out_W)] name:nil];
              MPSGraphTensor *weights1 = [mpsGraph reshapeTensor:weightTensor withShape:@[@(out_C),@(in_C/groups),@(1),@(1)] name:nil];

              MPSGraphTensor *x_grad = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:output_grad weightsTensor:weights1 outputShape:@[@(B),@(in_C),@(in_D),@(in_H*in_W)] forwardConvolutionDescriptor:conv2dDescriptor name: nil];

              gradInputTensor = [mpsGraph reshapeTensor:x_grad withShape:@[@(B),@(in_C),@(in_D),@(in_H),@(in_W)] name:nil];

          }
          else{
              MPSGraphTensor *output_grad = reshapePermuteReshape(mpsGraph, gradOutputTensor, @[@(B),@(out_C),@(out_D),@(out_H),@(out_W)], @[@0,@2,@1,@3,@4], @[@(B*out_D),@(out_C),@(out_H),@(out_W)]); //short-hand
              
              MPSGraphTensor *weights1 = [mpsGraph reshapeTensor:weightTensor withShape:@[@(out_C),@(in_C*k_D/groups),@(k_H),@(k_W)] name:nil];
              
              MPSGraphTensor *x2d_grad___ = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:output_grad weightsTensor:weights1 outputShape:@[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)] forwardConvolutionDescriptor:conv2dDescriptor name: nil];
              
              MPSGraphTensor *x2d_grad = reshapePermuteReshape(mpsGraph, x2d_grad___, @[@(B),@(out_D),@(in_C),@(k_D),@(in_H),@(in_W)], @[@0,@2,@3,@1,@4,@5], @[@(B*in_C),@(k_D),@(out_D),@(in_H*in_W)]); //short-hand
              
              MPSGraphTensor *x_grad = unfoldConvolution2d(mpsGraph, x2d_grad, stride[0], padding[0], dilation[0], k_D, getMPSScalarType(grad_output_t),@[@(B*in_C),@1,@(in_D),@(in_H*in_W)],true);
              
              gradInputTensor = [mpsGraph reshapeTensor:x_grad withShape:@[@(B),@(in_C),@(in_D),@(in_H),@(in_W)] name:nil];
          }
          
      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 at::MemoryFormat::Contiguous,
                                 groups);
        MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                                dimension:-3
                                                            withDimension:-4
                                                                     name:nil];
        gradInputTensor =
            [mpsGraph depthwiseConvolution3DDataGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                     weightsTensor:weightTransposeTensor
                                                                       outputShape:mps_input_shape
                                                                        descriptor:depthWiseConv3dDescriptor_
                                                                              name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradInputTensor = [mpsGraph convolution2DDataGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                          weightsTensor:weightTensor
                                                                            outputShape:mps_input_shape
                                                           forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                   name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->gradInputTensor_ = gradInputTensor;
    });

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t, gradOutputShape);
    auto weightsPlaceholder = Placeholder(cachedGraph->weightTensor_, weight_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, *grad_input);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, weightsPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
  return *grad_input;
}

static Tensor mps_convolution_backward_weights(IntArrayRef weight_size,
                                               const Tensor& grad_output_t,
                                               const Tensor& input_t,
                                               IntArrayRef padding,
                                               IntArrayRef stride,
                                               IntArrayRef dilation,
                                               int64_t groups,
                                               bool bias_defined) {
  using namespace at::native::mps;
  using namespace mps;
  bool is3DConv = input_t.dim() == 5;
  TORCH_CHECK(isFloatingType(grad_output_t.scalar_type()), "Convolution is supported only for Floating types");
  CheckedFrom c = "mps_convolution_backward_weights";
  auto memory_format = grad_output_t.suggest_memory_format();
  bool is_channels_last = (memory_format == at::MemoryFormat::ChannelsLast) && !is3DConv;

  MPSShape* gradOutputShape = mps::getMPSShape(grad_output_t, memory_format);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_output{grad_output_t, "grad_output", 1};
  TensorArg input{input_t, "input", 2};

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t =
      at::empty(weight_size, grad_output_t.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  TensorArg grad_weight{grad_weight_t, "result", 0};

  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
  };

  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();

    string mem_format_key;
    switch (memory_format) {
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
    string key;
    if (is3DConv) {
      key = "mps_3d_convolution_backward_weights:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(stride[2]) + ":" + std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" +
          std::to_string(dilation[2]) + ":" + std::to_string(padding[0]) + ":" + std::to_string(padding[1]) + ":" +
          std::to_string(padding[2]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, input_t, grad_weight_t}) + ":" + string([ns_shape_key UTF8String]);
    } else {
      key = "mps_convolution_backward_weights:" + std::to_string(stride[0]) + ":" + std::to_string(stride[1]) + ":" +
          std::to_string(dilation[0]) + ":" + std::to_string(dilation[1]) + ":" + std::to_string(padding[0]) + ":" +
          std::to_string(padding[1]) + ":" + std::to_string(groups) + ":" + mem_format_key +
          getTensorsStringKey({grad_output_t, input_t, grad_weight_t}) + ":" + string([ns_shape_key UTF8String]);
    }
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSShape* inputShape = mps::getMPSShape(input_t);
      bool isDepthwiseConv = ((groups > 1 && (mps_weight_shape[1].intValue == 1)) && inputShape.count >= 4 &&
                              mps_weight_shape.count >= 4 && !is_channels_last);

      MPSGraphTensor* gradOutputTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(grad_output_t), gradOutputShape);
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_t);

      MPSGraphTensor* gradOutputTensorTranspose = gradOutputTensor;
      if (is_channels_last) {
        gradOutputTensorTranspose = mps::convertNHWCtoNCHW(mpsGraph, gradOutputTensorTranspose);
      }

      MPSGraphTensor* gradWeightTensor;
      if (is3DConv) {
        
          const int64_t B = inputShape[0].intValue;
          const int64_t in_C = inputShape[1].intValue;
          const int64_t out_C = mps_weight_shape[0].intValue;
          
          const int64_t k_D = mps_weight_shape[2].intValue;
          const int64_t k_H = mps_weight_shape[3].intValue;
          const int64_t k_W = mps_weight_shape[4].intValue;
          const int64_t in_D = inputShape[2].intValue;
          const int64_t in_H = inputShape[3].intValue;
          const int64_t in_W = inputShape[4].intValue;

          const int64_t out_D = gradOutputShape[2].intValue;
          const int64_t out_H = gradOutputShape[3].intValue;
          const int64_t out_W = gradOutputShape[4].intValue;


          MPSGraphConvolution2DOpDescriptor* conv2dDescriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[2] strideInY:stride[1] dilationRateInX:dilation[2] dilationRateInY:dilation[1] groups:groups paddingLeft:padding[2] paddingRight:padding[2] paddingTop:padding[1] paddingBottom:padding[1] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

          MPSGraphTensor *weight_grad_;
          if((k_D==1)&(k_H==1)&(k_W==1)&(stride[0]==1)&(stride[1]==1)&(stride[2]==1)&(padding[0]==0)&(padding[1]==0)&(padding[2]==0)){ //1x1x1 convolution (aka linear layer) no permutation/unfold necessary
              MPSGraphTensor *input1 = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B),@(in_C),@(in_D),@(in_H*in_W)] name:nil];
              MPSGraphTensor *output_grad = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B),@(out_C),@(out_D),@(out_H*out_W)] name:nil];

              weight_grad_ = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:output_grad sourceTensor:input1 outputShape:@[@(out_C),@(in_C/groups),@(1),@(1)] forwardConvolutionDescriptor:conv2dDescriptor name:nil];
              
          }
          else{
              //unfold input tensor (same as forward conv3d)
              MPSGraphTensor *input1 = [mpsGraph reshapeTensor:inputTensor withShape:@[@(B*in_C),@1,@(in_D),@(in_H*in_W)] name:nil];
              MPSGraphTensor *unfold_ = unfoldConvolution2d(mpsGraph, input1, stride[0], padding[0], dilation[0], k_D, getMPSScalarType(input_t),nil,true); //2nd last arg nil = fwd, last arg true not transposed
              MPSGraphTensor *unfold = reshapePermuteReshape(mpsGraph, unfold_, @[@(B),@(in_C),@(k_D),@(out_D),@(in_H),@(in_W)], @[@0,@3,@1,@2,@4,@5], @[@(B*out_D),@(in_C*k_D),@(in_H),@(in_W)]); //short-hand
              
              MPSGraphTensor *output_grad = reshapePermuteReshape(mpsGraph, gradOutputTensor, @[@(B),@(out_C),@(out_D),@(out_H),@(out_W)], @[@0,@2,@1,@3,@4], @[@(B*out_D),@(out_C),@(out_H),@(out_W)]); //short-hand
                            
              weight_grad_ = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:output_grad sourceTensor:unfold outputShape:@[@(out_C),@(in_C*k_D/groups),@(k_H),@(k_W)] forwardConvolutionDescriptor:conv2dDescriptor name:nil];
          }
          gradWeightTensor = [mpsGraph reshapeTensor:weight_grad_ withShape:mps_weight_shape name:nil];


      } else if (isDepthwiseConv) {
        MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor_ =
            [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];
        fill_depthwise_conv_desc(depthWiseConv3dDescriptor_,
                                 stride[1],
                                 stride[0],
                                 dilation[1],
                                 dilation[0],
                                 padding[1],
                                 padding[0],
                                 at::MemoryFormat::Contiguous,
                                 groups);
        NSNumber* outputFeatChannelDim = mps_weight_shape[0];
        MPSShape* weightShapeTranspose = @[ @1, outputFeatChannelDim, mps_weight_shape[2], mps_weight_shape[3] ];
        MPSGraphTensor* gradWeightTensorTranspose =
            [mpsGraph depthwiseConvolution3DWeightsGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                         sourceTensor:inputTensor
                                                                          outputShape:weightShapeTranspose
                                                                           descriptor:depthWiseConv3dDescriptor_
                                                                                 name:nil];
        gradWeightTensor = [mpsGraph transposeTensor:gradWeightTensorTranspose dimension:-3 withDimension:-4 name:nil];
      } else {
        MPSGraphConvolution2DOpDescriptor* conv2dDescriptor_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
        fill_conv_desc(conv2dDescriptor_,
                       stride[1],
                       stride[0],
                       dilation[1],
                       dilation[0],
                       padding[1],
                       padding[0],
                       at::MemoryFormat::Contiguous,
                       groups);

        gradWeightTensor = [mpsGraph convolution2DWeightsGradientWithIncomingGradientTensor:gradOutputTensorTranspose
                                                                               sourceTensor:inputTensor
                                                                                outputShape:mps_weight_shape
                                                               forwardConvolutionDescriptor:conv2dDescriptor_
                                                                                       name:nil];
      }

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradWeightTensor_ = gradWeightTensor;
    });

    auto gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output_t, gradOutputShape);
    auto inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input_t);
    auto outputPlaceholder = Placeholder(cachedGraph->gradWeightTensor_, grad_weight_t);

    auto feeds = dictionaryFromPlaceholders(gradOutputPlaceholder, inputPlaceholder);
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return grad_weight_t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mps_convolution_backward(const at::Tensor& input,
                                                                        const at::Tensor& grad_output,
                                                                        const at::Tensor& weight,
                                                                        IntArrayRef padding,
                                                                        IntArrayRef stride,
                                                                        IntArrayRef dilation,
                                                                        int64_t groups,
                                                                        std::array<bool, 3> output_mask) {
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
      grad_input = mps_convolution_backward_input(
          input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1]) {
      grad_weight = mps_convolution_backward_weights(
          weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

static Tensor mps_convolution_transpose_forward(const Tensor& grad_output,
                                                const Tensor& weight,
                                                IntArrayRef padding,
                                                IntArrayRef output_padding,
                                                IntArrayRef stride,
                                                IntArrayRef dilation,
                                                int64_t groups) {
  auto input_size =
      conv_input_size(grad_output.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
    
    return mps_convolution_backward_input(input_size, grad_output, weight, padding, stride, dilation, groups, false);
    
}
Tensor _mps_convolution_transpose(const Tensor& input_t,
                                  const Tensor& weight_t,
                                  IntArrayRef padding,
                                  IntArrayRef output_padding,
                                  IntArrayRef stride,
                                  IntArrayRef dilation,
                                  int64_t groups) {
  //TORCH_CHECK(input_t.dim() < 5, "ConvTranspose 3D is not supported on MPS");
    bool is3DConv = input_t.dim() == 5;

    if(is3DConv){
        return _mps_convTrans3d_impl(input_t, weight_t, padding, output_padding, stride, dilation, groups);
    } else{
        return mps_convolution_transpose_forward(input_t, weight_t, padding, output_padding, stride, dilation, groups);
    }
}

static Tensor mps_convolution_transpose_backward_input(const Tensor& grad_output_t,
                                                       const Tensor& weight_t,
                                                       IntArrayRef padding,
                                                       IntArrayRef stride,
                                                       IntArrayRef dilation,
                                                       int64_t groups,
                                                       IntArrayRef input_shape) {
  return _mps_convolution_impl(grad_output_t, weight_t, std::nullopt, padding, stride, dilation, groups, input_shape);
}

static Tensor mps_convolution_transpose_backward_weight(IntArrayRef weight_size,
                                                        const Tensor& grad_output_t,
                                                        const Tensor& input_t,
                                                        IntArrayRef padding,
                                                        IntArrayRef stride,
                                                        IntArrayRef dilation,
                                                        int64_t groups) {
  return mps_convolution_backward_weights(
      weight_size, input_t, grad_output_t, padding, stride, dilation, groups, false);
}

std::tuple<Tensor, Tensor> mps_convolution_transpose_backward(const Tensor& input,
                                                              const Tensor& grad_output,
                                                              const Tensor& weight,
                                                              IntArrayRef padding,
                                                              IntArrayRef output_padding,
                                                              IntArrayRef stride,
                                                              IntArrayRef dilation,
                                                              int64_t groups,
                                                              std::array<bool, 2> output_mask) {
  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input =
        mps_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, input.sizes());
  }
  if (output_mask[1]) {
    grad_weight = mps_convolution_transpose_backward_weight(
        weight.sizes(), grad_output, input, padding, stride, dilation, groups);
  }

  return std::tuple<Tensor, Tensor>{grad_input, grad_weight};
}

} // namespace at::native
