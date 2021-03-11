#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>

@implementation MPSCNNNeuronOp

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
