#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

@implementation MPSCNNNeuronOp

+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13)) {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronHardSigmoid* neuron = nil;
  dispatch_once(&onceToken, ^{
    neuron = [[MPSCNNNeuronHardSigmoid alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
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
        initWithDevice:[MPSCNNContext sharedInstance].device
                     a:0];
  });
  return relu;
}

+ (MPSCNNNeuronSigmoid*)sigmoid {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronSigmoid* sigmoid = nil;
  dispatch_once(&onceToken, ^{
    sigmoid = [[MPSCNNNeuronSigmoid alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device];
  });
  return sigmoid;
}

+ (MPSCNNNeuronTanH*)tanh {
  static dispatch_once_t onceToken;
  static MPSCNNNeuronTanH* tanh = nil;
  dispatch_once(&onceToken, ^{
    tanh = [[MPSCNNNeuronTanH alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
                     a:1
                     b:1];
  });
  return tanh;
}

@end
