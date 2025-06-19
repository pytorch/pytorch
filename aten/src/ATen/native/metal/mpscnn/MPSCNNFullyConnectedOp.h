#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalNeuronType.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <Foundation/Foundation.h>

API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNFullyConnectedOp : NSObject<MPSCNNOp>
+ (MPSCNNFullyConnectedOp*)linear:(const at::native::metal::Conv2DParams&)params
                          weights:(float*)w
                             bias:(float*)b
                     neuronFilter:(at::native::metal::NeuronType)t;
@end
