#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalNeuronType.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <Foundation/Foundation.h>

using namespace at::native::metal;
API_AVAILABLE(ios(10.0), macos(10.13))
@interface MPSCNNFullyConnectedOp : NSObject<MPSCNNOp>
+ (MPSCNNFullyConnectedOp*)linear:(const Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(NeuronType)t;
@end
