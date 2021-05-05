#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
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
  /*
  `clamp(vector<half4>, float, float)` is not available on iOS 10.0,
  have to use `clamp(vector<half4>, half4, half4)` instead.
  */
  id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:at::native::metal::mpscnn::kernelFor(
                        _X, "clamp_half4", "clamp_half4_nonarray")];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[_X texture] atIndex:0];
  [encoder setTexture:[_Y texture] atIndex:1];
  id<MTLBuffer> clampBuffer = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:2 * sizeof(fp16_t)
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
  fp16_t* clampBufferPtr = (fp16_t*)[clampBuffer contents];
  clampBufferPtr[0] = _min.floatValue;
  clampBufferPtr[1] = _max.floatValue;
  [encoder setBuffer:clampBuffer offset:0 atIndex:0];
  const auto& launchParams =
      at::native::metal::mpscnn::spatialPointwiseKernelLaunchParams(state, _Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [_X markRead];
  [_Y markRead];
}

@end
