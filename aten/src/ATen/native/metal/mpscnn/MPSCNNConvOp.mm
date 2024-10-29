#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>

#include <c10/util/Exception.h>

@implementation MPSCNNConvDataSource {
  void* _weights;
  float* _bias;
  MPSCNNConvolutionDescriptor* _descriptor;
}

- (id)initWithWeights:(void*)weights
                 Bias:(float*)bias
                 Desc:(MPSCNNConvolutionDescriptor*)desc
    API_AVAILABLE(ios(11.0), macos(10.13)) {
  self = [super init];
  if (self) {
    _weights = (float*)weights;
    _bias = (float*)bias;
    _descriptor = desc;
  }
  return self;
}

- (nonnull id)copyWithZone:(nullable NSZone*)zone {
  MPSCNNConvDataSource* dataSource = [MPSCNNConvDataSource allocWithZone:zone];
  dataSource->_weights = _weights;
  dataSource->_bias = _bias;
  dataSource->_descriptor = _descriptor;
  return dataSource;
}

- (float* _Nullable)biasTerms {
  return _bias;
}

- (MPSDataType)dataType API_AVAILABLE(ios(11.0), macos(10.13)) {
  return MPSDataTypeFloat32;
}

- (NSString* _Nullable)label {
  return @"";
}

- (BOOL)load {
  return true;
}

- (void)purge {
  _bias = nullptr;
  _weights = nullptr;
}

- (void*)weights {
  return _weights;
}

- (MPSCNNConvolutionDescriptor* _Nonnull)descriptor {
  return _descriptor;
}

@end

@implementation MPSCNNConvOp {
}

@synthesize kernel = _kernel;

+ (MPSCNNConvOp*)conv2d:(const at::native::metal::Conv2DParams&)params
                weights:(float*)w
                   bias:(float*)b
           neuronFilter:(at::native::metal::NeuronType)t API_AVAILABLE(ios(11.0), macos(10.13)) {
  using namespace at::native::metal::mpscnn;
  TORCH_CHECK(
      params.DX == params.DY == 1, "Dilated convolution is not supported yet.");
  const NSUInteger oC = params.OC;
  const NSUInteger iC = params.C;
  const NSUInteger kH = params.KH;
  const NSUInteger kW = params.KW;
  MPSCNNNeuron* neuron = at::native::metal::neuron(t);
  MPSCNNConvolutionDescriptor* desc = nil;
  if (params.isDepthwise()) {
    if (@available(iOS 11.0, *)) {
      desc = [MPSCNNDepthWiseConvolutionDescriptor
          cnnConvolutionDescriptorWithKernelWidth:kW
                                     kernelHeight:kH
                             inputFeatureChannels:iC
                            outputFeatureChannels:oC];

      desc.groups = 1;
#if TARGET_OS_MACCATALYST
      desc.fusedNeuronDescriptor = at::native::metal::neuronDescriptor(t);
#else
      desc.neuron = neuron;
#endif
    } else {
      TORCH_CHECK(
          false,
          "MPSCNNDepthWiseConvolutionDescriptor is only available on iOS 11.0 and above");
    }
  } else {
    if (params.G > 1) {
      TORCH_CHECK(
          params.IC % 4 == 0,
          "MPSCNNConvolution requires number of input \
        channels in each group to be multiple of 4 for \
        group > 1.");
    }
    if (@available(iOS 11.0, *)) {
      desc = [MPSCNNConvolutionDescriptor
          cnnConvolutionDescriptorWithKernelWidth:kW
                                     kernelHeight:kH
                             inputFeatureChannels:iC
                            outputFeatureChannels:oC];
      desc.groups = params.G;
#if TARGET_OS_MACCATALYST
      desc.fusedNeuronDescriptor = at::native::metal::neuronDescriptor(t);
#else
      desc.neuron = neuron;
#endif
    } else {
      TORCH_CHECK(
          false,
          "MPSCNNConvolutionDescriptor is only available on iOS 11.0 and above");
    }
  }
  desc.strideInPixelsX = params.SX;
  desc.strideInPixelsY = params.SY;
  id<MPSCNNConvolutionDataSource> dataSource =
      [[MPSCNNConvDataSource alloc] initWithWeights:(float*)w
                                               Bias:(float*)b
                                               Desc:desc];
  MPSCNNConvolution* conv = nil;
  if (@available(iOS 11.0, *)) {
    conv = [[MPSCNNConvolution alloc]
        initWithDevice:[MetalContext sharedInstance].device
               weights:dataSource];

  } else {
    TORCH_CHECK(
        false, "MPSCNNConvolution is only available on iOS 11.0 and above");
  }
  [conv setEdgeMode:MPSImageEdgeModeZero];
  MPSOffset offset;
  offset.x = computeMPSAlignOffset(kW, params.PX);
  offset.y = computeMPSAlignOffset(kH, params.PY);
  offset.z = 0;
  [conv setOffset:offset];

  TORCH_CHECK(static_cast<int64_t>(conv.inputFeatureChannels) == params.IC * params.G);
  TORCH_CHECK(oC % conv.groups == 0);
  TORCH_CHECK(conv.outputFeatureChannels == oC);
  TORCH_CHECK(conv.kernelWidth == kW);
  TORCH_CHECK(conv.kernelHeight == kH);

  MPSCNNConvOp* op = [MPSCNNConvOp new];
  op->_kernel = conv;
  return op;
}

- (void)encode:(id<MTLCommandBuffer>)cb
         sourceImage:(MPSImage*)src
    destinationImage:(MPSImage*)dst {
  [_kernel encodeToCommandBuffer:cb sourceImage:src destinationImage:dst];
}

@end
