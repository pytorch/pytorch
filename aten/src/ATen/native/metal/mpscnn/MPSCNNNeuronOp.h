#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSCNNNeuronOp : NSObject

+ (MPSCNNNeuronReLU*)relu;
+ (MPSCNNNeuronSigmoid*)sigmoid;
+ (MPSCNNNeuronTanH*)tanh;

@end
