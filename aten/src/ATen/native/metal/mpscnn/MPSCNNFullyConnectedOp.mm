#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNFullyConnectedOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

@implementation MPSCNNFullyConnectedOp

@synthesize kernel = _kernel;

+ (MPSCNNFullyConnectedOp*)linear:(const Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(NeuronType)t
    API_AVAILABLE(ios(10.0), macos(10.13)) {
  MPSCNNNeuron* neuron = neuronType(t);
  MPSCNNConvolutionDescriptor* desc = [MPSCNNConvolutionDescriptor
      cnnConvolutionDescriptorWithKernelWidth:params.KW
                                 kernelHeight:params.KH
                         inputFeatureChannels:params.IC
                        outputFeatureChannels:params.OC
                                 neuronFilter:neuron];
  desc.strideInPixelsX = 1;
  desc.strideInPixelsY = 1;

  MPSCNNFullyConnected* fc = nil;
  if (@available(iOS 11.0, *)) {
    MPSCNNConvDataSource* ds =
        [[MPSCNNConvDataSource alloc] initWithWeights:(float*)w
                                                 Bias:(float*)b
                                                 Desc:desc];
    fc = [[MPSCNNFullyConnected alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
               weights:ds];
  } else {
#if TARGET_OS_IPHONE
    fc = [[MPSCNNFullyConnected alloc]
               initWithDevice:[MPSCNNContext sharedInstance].device
        convolutionDescriptor:desc
                kernelWeights:w
                    biasTerms:b
                        flags:MPSCNNConvolutionFlagsNone];
#endif
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
