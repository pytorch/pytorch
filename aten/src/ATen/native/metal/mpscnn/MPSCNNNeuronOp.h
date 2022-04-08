#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSCNNNeuronOp : NSObject

+ (MPSCNNNeuronHardSigmoid*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13));
+ (MPSCNNNeuronReLU*)relu;
+ (MPSCNNNeuronSigmoid*)sigmoid;
+ (MPSCNNNeuronTanH*)tanh;

@end

API_AVAILABLE(ios(11.3), macos(10.13), macCatalyst(13.0))
@interface MPSCNNNeuronOpDescriptor : NSObject

+ (MPSNNNeuronDescriptor*)hardSigmoidDescriptor;
+ (MPSNNNeuronDescriptor*)reluDescriptor;
+ (MPSNNNeuronDescriptor*)sigmoidDescriptor;
+ (MPSNNNeuronDescriptor*)tanhDescriptor;

@end
