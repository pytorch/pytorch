#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNFullyConnectedOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

@implementation MPSCNNFullyConnectedOp

@synthesize kernel = _kernel;

+ (MPSCNNFullyConnectedOp*)linear:(const at::native::metal::Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(at::native::metal::NeuronType)t
    API_AVAILABLE(ios(11.0), macos(10.13)) {
  MPSCNNNeuron* neuron = at::native::metal::neuron(t);
  MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
      cnnConvolutionDescriptorWithKernelWidth:params.KW
                                 kernelHeight:params.KH
                         inputFeatureChannels:params.IC
                        outputFeatureChannels:params.OC];
#if TARGET_OS_MACCATALYST
  desc.fusedNeuronDescriptor = at::native::metal::neuronDescriptor(t);
#else
  desc.neuron = neuron;
#endif
  desc.strideInPixelsX = 1;
  desc.strideInPixelsY = 1;

  MPSCNNFullyConnected* fc = nil;
  if (@available(iOS 11.0, *)) {
    MPSCNNConvDataSource* ds =
        [[MPSCNNConvDataSource alloc] initWithWeights:(float*)w
                                                 Bias:(float*)b
                                                 Desc:desc];
    fc = [[MPSCNNFullyConnected alloc]
        initWithDevice:[MetalContext sharedInstance].device
               weights:ds];
  } else {
    TORCH_CHECK(
        false,
        "MPSCNNFullyConnectedOp is only available on iOS 11.0 and above");
  }
  [fc setClipRect:MTLRegionMake3D(0, 0, 0, 1, 1, params.N)];
  [fc setOffset:{.x = static_cast<NSInteger>(params.W / 2),
                 .y = static_cast<NSInteger>(params.H / 2),
                 .z = 0}];
  MPSCNNFullyConnectedOp* kernel = [MPSCNNFullyConnectedOp new];
  kernel->_kernel = fc;
  return kernel;
}

- (void)encode:(id<MTLCommandBuffer>)cb
         sourceImage:(MPSImage*)src
    destinationImage:(MPSImage*)dst {
  [_kernel encodeToCommandBuffer:cb sourceImage:src destinationImage:dst];
}

@end
