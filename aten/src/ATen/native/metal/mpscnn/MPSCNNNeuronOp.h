#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSCNNNeuronOp : NSObject

+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13));
+ (MPSCNNNeuronReLU*)relu;
+ (MPSCNNNeuronSigmoid*)sigmoid;
+ (MPSCNNNeuronTanH*)tanh;

@end
