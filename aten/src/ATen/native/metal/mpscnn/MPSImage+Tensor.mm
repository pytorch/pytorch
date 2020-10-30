#include <ATen/native/metal/MetalUtils.h>
#include <ATen/native/metal/mpscnn/MPSCNN.h>
#include <ATen/native/metal/mpscnn/MPSCNNContext.h>
#include <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

#include <torch/script.h>

using namespace at::native;
@implementation MPSImage (Tensor)

+ (MPSImage*)imageFromCPUTensor:(const at::Tensor&)tensor {
  TORCH_CHECK(tensor.device().is_cpu());
  TORCH_CHECK(tensor.dim() == 4);
  auto contiguousTensor = tensor.contiguous();
  float* src = tensor.data_ptr<float>();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  auto c4 = metal::NCHW_to_NC4(src, sizes);
  auto c4fp16 = metal::fp32_to_fp16(c4);
  return [self imageFromFp16Array:c4fp16.data() Sizes:sizes];
}

+ (MPSImage*)imageFromFp16Array:(const uint16_t*)src
                          Sizes:(const std::vector<int64_t>&)sizes {
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
      [[MPSImage alloc] initWithDevice:[MPSCNNContext sharedInstance].device
                       imageDescriptor:desc];

  int64_t slices = (C + 3) / 4 * N;
  int64_t numComponents = image.featureChannels < 3 ? image.featureChannels : 4;
  int64_t bytesPerRow = W * numComponents * sizeof(uint16_t);
  uint8_t* ptr = (uint8_t*)src;
  for (int i = 0; i < slices; ++i) {
    [image.texture replaceRegion:MTLRegionMake2D(0, 0, W, H)
                     mipmapLevel:0
                           slice:i
                       withBytes:ptr
                     bytesPerRow:bytesPerRow
                   bytesPerImage:0];
    ptr += H * bytesPerRow;
  }
  return image;
}

+ (MPSImage*)imageFromSize:(const std::vector<int64_t>&)size {
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:size[3]
                                height:size[2]
                       featureChannels:size[1]
                        numberOfImages:size[0]
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  return [[MPSImage alloc] initWithDevice:[MPSCNNContext sharedInstance].device
                          imageDescriptor:desc];
}

- (std::vector<uint16_t>)toFp16Array {
  if (self.pixelFormat == MTLPixelFormatR16Float ||
      self.pixelFormat == MTLPixelFormatRG16Float ||
      self.pixelFormat == MTLPixelFormatRGBA16Float) {
    int64_t slices = (self.featureChannels + 3) / 4;
    int64_t C = self.featureChannels < 3 ? self.featureChannels : slices * 4;
    int64_t numComponents = self.featureChannels < 3 ? self.featureChannels : 4;
    int64_t count = self.width * self.height * self.numberOfImages * C;
    std::vector<uint16_t> output(count, 0);
    int64_t bytesPerRow = self.width * numComponents * sizeof(uint16_t);
    uint8_t* buffer = (uint8_t*)output.data();
    for (int i = 0; i < slices * self.numberOfImages; ++i) {
      [self.texture getBytes:buffer
                 bytesPerRow:bytesPerRow
               bytesPerImage:0
                  fromRegion:MTLRegionMake2D(0, 0, self.width, self.height)
                 mipmapLevel:0
                       slice:i];
      buffer += self.height * bytesPerRow;
    }
    return output;
  }
  TORCH_CHECK(
      false, "Copy to float buffer failed: The pixel format didn't match");
  return {};
}

- (at::Tensor)toCPUTensor {
  auto outputSize = [self sizes];
  std::vector<uint16_t> fp16 = [self toFp16Array];
  auto fp32 = metal::fp16_to_fp32(fp16);
  std::vector<float> fp32_nchw = metal::NC4_to_NCHW(fp32.data(), outputSize);
  auto tensor = at::empty(outputSize);
  int64_t size_bytes = at::prod_intlist(outputSize) * sizeof(float);
  memcpy(tensor.data_ptr(), fp32_nchw.data(), size_bytes);
  return tensor;
}

- (std::vector<int64_t>)sizes {
  int64_t N = self.numberOfImages;
  int64_t C = self.featureChannels;
  int64_t H = self.height;
  int64_t W = self.width;
  return {N, C, H, W};
}

+ (MPSTemporaryImage*)temporaryImageFromSize:(const std::vector<int64_t>&)size
                               commandBuffer:(MetalCommandBuffer*)cmdBuffer {
  NSCAssert(cmdBuffer, @"CommandBuffer is nil!");
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:size[3]
                                height:size[2]
                       featureChannels:size[1]
                        numberOfImages:size[0]
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  MPSTemporaryImage* image =
      [MPSTemporaryImage temporaryImageWithCommandBuffer:cmdBuffer.buffer
                                         imageDescriptor:desc];
  image.readCount = INT_MAX;
  [cmdBuffer add:image];
  return image;
}

- (BOOL)isTemporaryImage {
  return [self isKindOfClass:[MPSTemporaryImage class]];
}

- (void)markRead {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    if (tmpImage.readCount > 0) {
      tmpImage.readCount -= 1;
    }
  }
}

- (void)recycle {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    if (tmpImage.readCount > 0) {
      tmpImage.readCount = 0;
    }
  }
}

- (int64_t)readCount {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    return (int64_t)tmpImage.readCount;
  }
  return -1;
}

@end

@implementation MPSImage (Shaders)

+ (MPSImage*)imageFromImage:(MPSImage*)X {
  auto&& sizes = [X sizes];
  MPSImage* Y = [MPSImage imageFromSize:sizes];
  MetalCommandBuffer* cb = [MetalCommandBuffer newBuffer];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:metal::mpscnn::kernelFor(X, @"copy", @"copy_nonarray")];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [cb synchronize];
  return Y;
}

+ (MPSTemporaryImage*)temporaryImageFromImage:(MPSImage*)X
                                CommandBuffer:(MetalCommandBuffer*)cb {
  NSCAssert(cb, @"CommandBuffer is nil!");
  MPSTemporaryImage* Y = [MPSImage temporaryImageFromSize:[X sizes]
                                            commandBuffer:cb];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:metal::mpscnn::kernelFor(X, @"copy", @"copy_nonarray")];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  return Y;
}

+ (MPSImage*)imageFromTemporaryImage:(MPSTemporaryImage*)X
                       CommandBuffer:(MetalCommandBuffer*)cb
                  waitUntilCompleted:(BOOL)b {
  NSCAssert(cb, @"CommandBuffer is nil!");
  auto&& sizes = [X sizes];
  MPSImage* Y = [MPSImage imageFromSize:sizes];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:metal::mpscnn::kernelFor(X, @"copy", @"copy_nonarray")];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X markRead];
  if (b) {
    [cb synchronize];
  }
  return Y;
}

+ (MPSImage*)imageFromHost:(const float*)src
                     Sizes:(const std::vector<int64_t>&)sizes {
  int64_t size_bytes = at::prod_intlist(sizes) * sizeof(float);
  // allocte buffer on CPU
  id<MTLBuffer> buff = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
  memcpy(buff.contents, src, size_bytes);
  MPSImage* output = [MPSImage imageFromSize:sizes];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   output,
                                   @"copy_nchw_to_metal",
                                   @"copy_nchw_to_metal_nonarray")
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
  [cb synchronize];
  return output;
}

+ (MPSTemporaryImage*)temporaryImageFromHost:(const float*)src
                                       Sizes:(const std::vector<int64_t>&)sizes
                               CommandBuffer:(MetalCommandBuffer*)cb {
  NSCAssert(cb, @"CommandBuffer is nil!");
  int64_t size_bytes = at::prod_intlist(sizes) * sizeof(float);
  // allocte buffer on CPU
  id<MTLBuffer> buff = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
  memcpy(buff.contents, src, size_bytes);
  MPSTemporaryImage* output = [MPSImage temporaryImageFromSize:sizes
                                                 commandBuffer:cb];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   output,
                                   @"copy_nchw_to_metal",
                                   @"copy_nchw_to_metal_nonarray")
                     Constants:@[
                       @(output.featureChannels),
                       @(output.height),
                       @(output.width)
                     ]];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
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

+ (void)copyToHost:(float*)dst FromImage:(MPSImage*)image {
  auto&& sizes = [image sizes];
  int64_t size_bytes = at::prod_intlist(sizes) * sizeof(float);
  id<MTLBuffer> buffer = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeDefault];

  id<MTLCommandBuffer> cb =
      [MPSCNNContext sharedInstance].commandQueue.commandBuffer;
  id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   image,
                                   @"copy_metal_to_nchw",
                                   @"copy_metal_to_nchw_nonarray")
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

@end
