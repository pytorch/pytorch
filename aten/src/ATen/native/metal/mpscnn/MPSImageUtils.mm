#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {

MPSImage* createStaticImage(const std::vector<int64_t>& sizes) {
  MPSImageDescriptor* desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:sizes[3]
                                height:sizes[2]
                       featureChannels:sizes[1]
                        numberOfImages:sizes[0]
                                 usage:MTLTextureUsageShaderRead |
                                 MTLTextureUsageShaderWrite];
  return [[MPSImage alloc] initWithDevice:[MPSCNNContext sharedInstance].device
                          imageDescriptor:desc];
}

MPSImage* createStaticImage(
    const uint16_t* src,
    const std::vector<int64_t>& sizes) {
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

MPSImage* createStaticImage(
    const float* src,
    const std::vector<int64_t>& sizes) {
  int64_t size_bytes = c10::multiply_integers(sizes) * sizeof(float);
  id<MTLBuffer> buff = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
  memcpy(buff.contents, src, size_bytes);
  MPSImage* output = createStaticImage(sizes);
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

MPSImage* createStaticImage(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu());
  TORCH_CHECK(tensor.dim() == 4);
  auto contiguousTensor = tensor.contiguous();
  float* src = tensor.data_ptr<float>();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  auto c4 = NCHWToNC4(src, sizes);
  auto c4fp16 = Fp32ToFp16(c4);
  return createStaticImage(c4fp16.data(), sizes);
}

MPSImage* createStaticImage(MPSImage* image) {
  MPSImage* Y = createStaticImage([image sizes]);
  MetalCommandBuffer* cb = [MetalCommandBuffer newBuffer];
  id<MTLComputeCommandEncoder> encoder = [cb.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:mpscnn::kernelFor(image, @"copy", @"copy_nonarray")];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[image texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      mpscnn::spatialPointwiseKernelLaunchParams(state, image);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [cb synchronize];
  return Y;
}

MPSImage* createStaticImage(
    MPSTemporaryImage* image,
    MetalCommandBuffer* buffer,
    bool waitUntilCompleted) {
  TORCH_CHECK(buffer);
  MPSImage* Y = createStaticImage([image sizes]);
  id<MTLComputeCommandEncoder> encoder = [buffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:mpscnn::kernelFor(image, @"copy", @"copy_nonarray")];

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
    [buffer synchronize];
  }
  return Y;
}

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const std::vector<int64_t>& sizes) {
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

MPSTemporaryImage* createTemporaryImage(
    MetalCommandBuffer* buffer,
    const std::vector<int64_t>& sizes,
    const float* src) {
  TORCH_CHECK(buffer);
  int64_t size_bytes = c10::multiply_integers(sizes) * sizeof(float);
  id<MTLBuffer> buff = [[MPSCNNContext sharedInstance].device
      newBufferWithLength:size_bytes
                  options:MTLResourceOptionCPUCacheModeWriteCombined];
  memcpy(buff.contents, src, size_bytes);
  MPSTemporaryImage* output = createTemporaryImage(buffer, sizes);
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
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:metal::mpscnn::kernelFor(image, @"copy", @"copy_nonarray")];
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

std::vector<uint16_t> staticImageToFp16Array(MPSImage* image) {
  if (image.pixelFormat == MTLPixelFormatR16Float ||
      image.pixelFormat == MTLPixelFormatRG16Float ||
      image.pixelFormat == MTLPixelFormatRGBA16Float) {
    int64_t slices = (image.featureChannels + 3) / 4;
    int64_t C = image.featureChannels < 3 ? image.featureChannels : slices * 4;
    int64_t numComponents =
        image.featureChannels < 3 ? image.featureChannels : 4;
    int64_t count = image.width * image.height * image.numberOfImages * C;
    std::vector<uint16_t> output(count, 0);
    int64_t bytesPerRow = image.width * numComponents * sizeof(uint16_t);
    uint8_t* buffer = (uint8_t*)output.data();
    for (int i = 0; i < slices * image.numberOfImages; ++i) {
      [image.texture getBytes:buffer
                  bytesPerRow:bytesPerRow
                bytesPerImage:0
                   fromRegion:MTLRegionMake2D(0, 0, image.width, image.height)
                  mipmapLevel:0
                        slice:i];
      buffer += image.height * bytesPerRow;
    }
    return output;
  }
  TORCH_CHECK(
      false, "Copy to float buffer failed: The pixel format didn't match");
}

at::Tensor staticImageToTensor(MPSImage* image) {
  auto outputSize = [image sizes];
  std::vector<uint16_t> fp16 = staticImageToFp16Array(image);
  auto fp32 = metal::Fp16ToFp32(fp16);
  std::vector<float> fp32_nchw = metal::NC4ToNCHW(fp32.data(), outputSize);
  auto tensor = at::empty(outputSize);
  int64_t size_bytes = c10::multiply_integers(outputSize) * sizeof(float);
  memcpy(tensor.data_ptr(), fp32_nchw.data(), size_bytes);
  return tensor;
}

}
}
}
