#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
C10_DIAGNOSTIC_POP()

@interface MPSCNNNeuronOp : NSObject

+ (MPSCNNNeuron*)hardSigmoid API_AVAILABLE(ios(11.0), macos(10.13));
+ (MPSCNNNeuron*)relu;
+ (MPSCNNNeuron*)sigmoid;
+ (MPSCNNNeuron*)tanh;

@end

API_AVAILABLE(ios(11.3), macos(10.13), macCatalyst(13.0))
@interface MPSCNNNeuronOpDescriptor : NSObject

+ (MPSNNNeuronDescriptor*)hardSigmoidDescriptor;
+ (MPSNNNeuronDescriptor*)reluDescriptor;
+ (MPSNNNeuronDescriptor*)sigmoidDescriptor;
+ (MPSNNNeuronDescriptor*)tanhDescriptor;

@end
