#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {

MPSImage* createStaticImage(IntArrayRef sizes) {
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:W
                                height:H
                       featureChannels:C
                        numberOfImages:N
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  MPSImage* image =
      [[MPSImage alloc] initWithDevice:[MetalContext sharedInstance].device
                       imageDescriptor:desc];
  image.label = [NSString
      stringWithFormat:@"[%d, %d, %d, %d]", (int)N, (int)C, (int)H, (int)W];
  return image;
}

MPSImage* createStaticImage(const float* src, IntArrayRef sizes) {
  int64_t size_bytes = c10::multiply_integers(sizes) * sizeof(float);
  id<MTLBuffer> buff = [[MetalContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceCPUCacheModeWriteCombined];
  memcpy(buff.contents, src, size_bytes);
  MPSImage* output = createStaticImage(sizes);
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   output,
                                   "copy_nchw_to_metal",
                                   "copy_nchw_to_metal_nonarray")
                     Constants:@[
                       @(output.featureChannels),
                       @(output.height),
                       @(output.width)
                     ]];
  MetalCommandBuffer* cb = [MetalCommandBuffer newBuffer];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setBuffer:buff offset:0 atIndex:0];
  [encoder setTexture:[output texture] atIndex:0];
  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, output);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [cb commit];
  return output;
}

MPSImage* createStaticImage(
    MPSTemporaryImage* image,
    MetalCommandBuffer* buffer,
    bool waitUntilCompleted) {
  TORCH_CHECK(buffer);
  MPSImage* Y = createStaticImage([image sizes]);
  id<MTLComputeCommandEncoder> encoder = [buffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      pipelineState:mpscnn::kernelFor(image, "copy", "copy_nonarray")];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[image texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, image);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  if (waitUntilCompleted) {
    [buffer commit];
  }
  return Y;
}

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    IntArrayRef sizes) {
  TORCH_CHECK(buffer);
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:W
                                height:H
                       featureChannels:C
                        numberOfImages:N
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  MPSTemporaryImage* image =
      [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer.buffer
                                         imageDescriptor:desc];
  image.readCount = INT_MAX;
  image.label = [NSString
      stringWithFormat:@"[%d, %d, %d, %d]", (int)N, (int)C, (int)H, (int)W];
  [buffer add:image];
  return image;
}

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    IntArrayRef sizes,
    const float* src) {
  TORCH_CHECK(buffer);
  int64_t size_bytes = c10::multiply_integers(sizes) * sizeof(float);
  id<MTLBuffer> buff = [[MetalContext sharedInstance].device
      newBufferWithBytes:src
                  length:size_bytes
                 options:MTLResourceStorageModeShared];
  MPSTemporaryImage* output = createTemporaryImage(buffer, sizes);
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   output,
                                   "copy_nchw_to_metal",
                                   "copy_nchw_to_metal_nonarray")
                     Constants:@[
                       @(output.featureChannels),
                       @(output.height),
                       @(output.width)
                     ]];
  id<MTLComputeCommandEncoder> encoder = [buffer.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setBuffer:buff offset:0 atIndex:0];
  [encoder setTexture:[output texture] atIndex:0];
  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, output);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  return output;
}

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    MPSImage* image) {
  TORCH_CHECK(buffer);
  MPSTemporaryImage* Y = createTemporaryImage(buffer, [image sizes]);
  id<MTLComputeCommandEncoder> encoder = [buffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      pipelineState:metal::mpscnn::kernelFor(image, "copy", "copy_nonarray")];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[image texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, image);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  return Y;
}

void copyImageToFloatBuffer(float* dst, MPSImage* image) {
  int64_t size_bytes = c10::multiply_integers([image sizes]) * sizeof(float);
  id<MTLBuffer> buffer = [[MetalContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceCPUCacheModeDefaultCache];

  id<MTLCommandBuffer> cb =
      [MetalContext sharedInstance].commandQueue.commandBuffer;
  id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   image,
                                   "copy_metal_to_nchw",
                                   "copy_metal_to_nchw_nonarray")
                     Constants:@[
                       @(image.featureChannels),
                       @(image.height),
                       @(image.width)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setBuffer:buffer offset:0 atIndex:0];
  [encoder setTexture:[image texture] atIndex:0];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, image);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  memcpy(dst, buffer.contents, buffer.length);
}

void copyImageToMetalBuffer(
    MetalCommandBuffer* cmdBuffer,
    id<MTLBuffer> dst,
    MPSImage* image) {
  TORCH_CHECK(cmdBuffer.buffer);
  id<MTLComputeCommandEncoder> encoder =
      [cmdBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   image,
                                   "copy_metal_to_nchw",
                                   "copy_metal_to_nchw_nonarray")
                     Constants:@[
                       @(image.featureChannels),
                       @(image.height),
                       @(image.width)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setBuffer:dst offset:0 atIndex:0];
  [encoder setTexture:[image texture] atIndex:0];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, image);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
}

}
}
}
