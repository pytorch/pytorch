#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalNeuronType.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <Foundation/Foundation.h>
C10_DIAGNOSTIC_POP()

API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNFullyConnectedOp : NSObject<MPSCNNOp>
+ (MPSCNNFullyConnectedOp*)linear:(const at::native::metal::Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(at::native::metal::NeuronType)t;
@end
