#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

#include <c10/macros/Macros.h>

C10_CLANG_DIAGNOSTIC_PUSH()
C10_CLANG_DIAGNOSTIC_IGNORE("-Wdeprecated-declarations")

@implementation MPSCNNNeuronOp

+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13)) {
// Remove this once we support iOS 11.3
#if TARGET_OS_MACCATALYST
  return nil;
#else
  static dispatch_once_t onceToken;
  static MPSCNNNeuronHardSigmoid* neuron = nil;
  dispatch_once(&onceToken, ^{
    neuron = [[MPSCNNNeuronHardSigmoid alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:1.0 / 6.0
                     b:0.5];
  });
  return neuron;
#endif
}

+ (MPSCNNNeuronReLU*)relu {
// Remove this once we support iOS 11.3
#if TARGET_OS_MACCATALYST
  return nil;
#else
  static MPSCNNNeuronReLU* relu = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    relu = [[MPSCNNNeuronReLU alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:0];
  });
  return relu;
#endif
}

+ (MPSCNNNeuronSigmoid*)sigmoid {
// Remove this once we support iOS 11.3
#if TARGET_OS_MACCATALYST
  return nil;
#else
  static dispatch_once_t onceToken;
  static MPSCNNNeuronSigmoid* sigmoid = nil;
  dispatch_once(&onceToken, ^{
    sigmoid = [[MPSCNNNeuronSigmoid alloc]
        initWithDevice:[MetalContext sharedInstance].device];
  });
  return sigmoid;
#endif
}

+ (MPSCNNNeuronTanH*)tanh {
// Remove this once we support iOS 11.3
#if TARGET_OS_MACCATALYST
  return nil;
#else
  static dispatch_once_t onceToken;
  static MPSCNNNeuronTanH* tanh = nil;
  dispatch_once(&onceToken, ^{
    tanh = [[MPSCNNNeuronTanH alloc]
        initWithDevice:[MetalContext sharedInstance].device
                     a:1
                     b:1];
  });
  return tanh;
#endif
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
    neuronDesc =
        [MPSNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeTanH
                                                         a:1.0
                                                         b:1.0];
  });
  return neuronDesc;
}

@end
