#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

#include <c10/macros/Macros.h>

C10_CLANG_DIAGNOSTIC_PUSH()
C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-declarations")

@implementation MPSCNNNeuronOp

+ (MPSCNNNeuron*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13)) {
  static MPSCNNNeuron* neuron = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
#if TARGET_OS_MACCATALYST
    neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalContext sharedInstance].device neuronDescriptor:[MPSCNNNeuronOpDescriptor hardSigmoidDescriptor]];
#else
    neuron = [[MPSCNNNeuronHardSigmoid alloc]
              initWithDevice:[MetalContext sharedInstance].device
              a:1.0 / 6.0
              b:0.5];
#endif
  });
  return neuron;
}

+ (MPSCNNNeuron*)relu {
  static MPSCNNNeuron* neuron = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
#if TARGET_OS_MACCATALYST
    neuron = [[MPSCNNNeuron alloc]
              initWithDevice:[MetalContext sharedInstance].device
              neuronDescriptor:[MPSCNNNeuronOpDescriptor reluDescriptor]];
#else
    neuron = [[MPSCNNNeuronReLU alloc]
              initWithDevice:[MetalContext sharedInstance].device
              a:0];
#endif
  });
  return neuron;
}

+ (MPSCNNNeuron*)sigmoid {
  static MPSCNNNeuron* neuron = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
#if TARGET_OS_MACCATALYST
    neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalContext sharedInstance].device neuronDescriptor:[MPSCNNNeuronOpDescriptor sigmoidDescriptor]];
#else
    neuron = [[MPSCNNNeuronSigmoid alloc]
              initWithDevice:[MetalContext sharedInstance].device];
#endif
  });
  return neuron;
}

+ (MPSCNNNeuron*)tanh {
  static MPSCNNNeuron* neuron = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
#if TARGET_OS_MACCATALYST
    neuron = [[MPSCNNNeuron alloc] initWithDevice:[MetalContext sharedInstance].device neuronDescriptor:[MPSCNNNeuronOpDescriptor tanhDescriptor]];
#else
    neuron = [[MPSCNNNeuronTanH alloc]
              initWithDevice:[MetalContext sharedInstance].device
              a:1
              b:1];
#endif
  });
  return neuron;
}

@end

C10_CLANG_DIAGNOSTIC_POP()

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
    neuronDesc = [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH
                                                                  a:1.0
                                                                  b:1.0];
  });
  return neuronDesc;
}

@end
