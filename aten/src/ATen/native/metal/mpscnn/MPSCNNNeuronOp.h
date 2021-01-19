#import <ATen/native/metal/MetalConvolution.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

using namespace at::native::metal;
@interface MPSCNNNeuronOp : NSObject

+ (MPSCNNNeuronReLU*)relu;
+ (MPSCNNNeuronSigmoid*)sigmoid;
+ (MPSCNNNeuronTanH*)tanh;

@end
