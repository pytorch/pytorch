#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalNeuronType.h>
#import <ATen/native/metal/mpscnn/MPSCNNOp.h>
#import <Foundation/Foundation.h>

API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNConvDataSource : NSObject<MPSCNNConvolutionDataSource>
@property(nonatomic, assign) void* weights;
@property(nonatomic, assign) float* bias;

- (id)initWithWeights:(void*)weights
                 Bias:(float*)bias
                 Desc:(MPSCNNConvolutionDescriptor*)desc;

@end

using namespace at::native::metal;
API_AVAILABLE(ios(11.0), macos(10.13))
@interface MPSCNNConvOp : NSObject<MPSCNNOp>
+ (MPSCNNConvOp*)conv2d:(const Conv2DParams&)params
                weights:(float*)w
                   bias:(float*)b
           neuronFilter:(NeuronType)t;
@end
