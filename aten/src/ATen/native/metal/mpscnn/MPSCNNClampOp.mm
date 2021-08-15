#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

@implementation MPSCNNClampOp {
  MPSImage* _X;
  MPSImage* _Y;
  NSNumber* _min;
  NSNumber* _max;
}

+ (id<MPSCNNShaderOp>)newWithTextures:(NSArray<MPSImage*>*)textures
                                 Args:(NSArray<NSNumber*>*)args {
  MPSCNNClampOp* op = [MPSCNNClampOp new];
  op->_X = textures[0];
  op->_Y = textures[1];
  op->_min = args[0];
  op->_max = args[1];

  return op;
}

- (void)encode:(id<MTLCommandBuffer>)cb {
  id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
  id<MTLComputePipelineState> state =
      [[MetalContext sharedInstance] specializedPipelineState:"clamp"
                                                    Constants:@[
                                                      @(_min.floatValue),
                                                      @(_max.floatValue),
                                                      @(_X.featureChannels),
                                                      @(_X.numberOfImages)
                                                    ]];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[_X texture] atIndex:0];
  [encoder setTexture:[_Y texture] atIndex:1];
  const auto& launchParams =
      at::native::metal::mpscnn::spatialPointwiseKernelLaunchParams(state, _Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
}

@end
