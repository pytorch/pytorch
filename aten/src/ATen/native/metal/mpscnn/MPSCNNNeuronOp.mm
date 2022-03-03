#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

@implementation MPSCNNNeuronOp

+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13)) {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronHardSigmoid* neuron = nil;
  dispatch_once(&onceToken, ^{
    neuron = [[MPSCNNNeuronHardSigmoid alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:1.0 / 6.0
                     b:0.5];
  });
  return neuron;
}

+ (MPSCNNNeuronReLU*)relu {
  static MPSCNNNeuronReLU* relu = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    relu = [[MPSCNNNeuronReLU alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:0];
  });
  return relu;
}

+ (MPSCNNNeuronSigmoid*)sigmoid {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronSigmoid* sigmoid = nil;
  dispatch_once(&onceToken, ^{
    sigmoid = [[MPSCNNNeuronSigmoid alloc]
        initWithDevice:[MetalContext sharedInstance].device];
  });
  return sigmoid;
}

+ (MPSCNNNeuronTanH*)tanh {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronTanH* tanh = nil;
  dispatch_once(&onceToken, ^{
    tanh = [[MPSCNNNeuronTanH alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:1
                     b:1];
  });
  return tanh;
}

@end

API_AVAILABLE(ios(11.3), macos(10.13), macCatalyst(13.0))
@implementation MPSCNNNeuronOpDescriptor

+ (MPSNNNeuronDescriptor*)hardSigmoidDescriptor {
  static dispatch_once_t onceToken;
  static MPSNNNeuronDescriptor* neuronDesc = nil;
  dispatch_once(&onceToken, ^{
    neuronDesc = [MPSNNNeuronDescriptor
        cnnNeuronDescriptorWithType:MPSCNNNeuronTypeHardSigmoid
                                  a:1.0 / 6.0
                                  b:0.5];
  });
  return neuronDesc;
}

+ (MPSNNNeuronDescriptor*)reluDescriptor {
  static dispatch_once_t onceToken;
  static MPSNNNeuronDescriptor* neuronDesc = nil;
  dispatch_once(&onceToken, ^{
    neuronDesc =
        [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU
                                                         a:0];
  });
  return neuronDesc;
}

+ (MPSNNNeuronDescriptor*)sigmoidDescriptor {
  static dispatch_once_t onceToken;
  static MPSNNNeuronDescriptor* neuronDesc = nil;
  dispatch_once(&onceToken, ^{
    neuronDesc = [MPSNNNeuronDescriptor
        cnnNeuronDescriptorWithType:MPSCNNNeuronTypeSigmoid];
  });
  return neuronDesc;
}

+ (MPSNNNeuronDescriptor*)tanhDescriptor {
  static dispatch_once_t onceToken;
  static MPSNNNeuronDescriptor* neuronDesc = nil;
  dispatch_once(&onceToken, ^{
    neuronDesc =
        [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH
                                                         a:1.0
                                                         b:1.0];
  });
  return neuronDesc;
}

@end
