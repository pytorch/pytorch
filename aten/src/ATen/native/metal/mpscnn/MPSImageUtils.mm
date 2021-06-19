#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {

MPSImage* createStaticImage(IntArrayRef sizes) {
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:sizes[3]
                                height:sizes[2]
                       featureChannels:sizes[1]
                        numberOfImages:sizes[0]
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  return [[MPSImage alloc] initWithDevice:[MetalContext sharedInstance].device
                          imageDescriptor:desc];
}

MPSImage* createStaticImage(const float* src, IntArrayRef sizes) {
  int64_t size_bytes = c10::multiply_integers(sizes) * sizeof(float);
  id<MTLBuffer> buff = [[MetalContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
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
  [image markRead];
  if (waitUntilCompleted) {
    [buffer commit];
  }
  return Y;
}

MPSTemporaryImage* createTemporaryImage(MetalCommandBuffer* buffer, IntArrayRef sizes) {
  TORCH_CHECK(buffer);
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:sizes[3]
                                height:sizes[2]
                       featureChannels:sizes[1]
                        numberOfImages:sizes[0]
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  MPSTemporaryImage* image =
      [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer.buffer
                                         imageDescriptor:desc];
  image.readCount = INT_MAX;
  [buffer add:image];
  return image;
}

MPSTemporaryImage* createTemporaryImage(MetalCommandBuffer* buffer, IntArrayRef sizes, const float* src) {
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
  [output markRead];
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

void copyToHost(float* dst, MPSImage* image) {
  int64_t size_bytes = c10::multiply_integers([image sizes]) * sizeof(float);
  id<MTLBuffer> buffer = [[MetalContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeDefault];

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

void copyToMetalBuffer(
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
