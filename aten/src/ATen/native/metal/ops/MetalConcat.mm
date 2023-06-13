
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

Tensor cat_batch(const Tensor& tensor, const ITensorListRef& tensors, MetalTensorImplStorage& mt) {
  MetalCommandBuffer* commandBuffer = getCommandBuffer(tensor);
  MPSImage* Y = mt.texture()->image();
  ushort cat_dim4_pointer = 0;
  for (const auto& t : tensors) {
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = getCommandBuffer(t);
    TORCH_CHECK(
        [commandBuffer isEqual:Xcb],
        @"inputs have different Metal command buffers");
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
        pipelineState:mpscnn::kernelFor(
                          X, "copy_offset", "copy_offset_nonarray")];
    id<MTLBuffer> offsetBuffer = [[MetalContext sharedInstance].device
        newBufferWithLength:1 * sizeof(ushort)
                    options:MTLResourceCPUCacheModeWriteCombined];
    ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
    offsetBufferPtr[0] = cat_dim4_pointer;

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    [encoder setBuffer:offsetBuffer offset:0 atIndex:0];

    const auto& launchParams =
        mpscnn::spatialPointwiseKernelLaunchParams(state, X);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    cat_dim4_pointer += t.size(0) * ((t.size(1) + 3) / 4);
  }
  auto output = makeTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat_feature(const Tensor& tensor, const ITensorListRef& tensors, MetalTensorImplStorage& mt) {
  MetalCommandBuffer* commandBuffer = getCommandBuffer(tensor);
  MPSImage* Y = mt.texture()->image();
  ushort channel_offset = 0;

  auto temp_size = tensor.sizes().vec();
  temp_size[1] = 4;
  MetalTensorImplStorage tt{temp_size};
  tt.texture()->setCommandBuffer(commandBuffer);
  tt.texture()->allocateTemporaryStorage(temp_size, commandBuffer);
  MPSImage* T = tt.texture()->image();

  for (const auto& t : tensors) {
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = getCommandBuffer(t);
    TORCH_CHECK(
        [commandBuffer isEqual:Xcb],
        @"inputs have different Metal command buffers");
    ushort tex_offset = channel_offset % 4;
    std::string kernelString = tex_offset == 0 ? "append_features" : "append_features_off";

    {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer.buffer computeCommandEncoder];
      id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
          specializedPipelineState:kernelString
                         Constants:@[
                           @(T.height),
                           @(T.width),
                           @(T.featureChannels),
                           @(T.numberOfImages),
                           @(X.height),
                           @(X.width),
                           @(X.featureChannels),
                           @(X.numberOfImages),
                         ]];
      id<MTLBuffer> offsetBuffer = [[MetalContext sharedInstance].device
          newBufferWithLength:6 * sizeof(ushort)
                      options:MTLResourceCPUCacheModeWriteCombined];
      ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
      offsetBufferPtr[0] = (X.featureChannels + tex_offset + 3) / 4;
      offsetBufferPtr[1] = (Y.featureChannels + 3) / 4;
      offsetBufferPtr[2] = channel_offset / 4;
      offsetBufferPtr[3] = (X.featureChannels + 3) / 4;
      offsetBufferPtr[4] = X.numberOfImages * offsetBufferPtr[0];
      offsetBufferPtr[5] = tex_offset;

      [encoder setComputePipelineState:state];
      if (tex_offset == 0) {
        [encoder setTexture:[X texture] atIndex:0];
        [encoder setTexture:[Y texture] atIndex:1];
        [encoder setBuffer:offsetBuffer offset:0 atIndex:0];
      }
      else {
        [encoder setTexture:[X texture] atIndex:0];
        [encoder setTexture:[T texture] atIndex:1];
        [encoder setTexture:[Y texture] atIndex:2];
        [encoder setBuffer:offsetBuffer offset:0 atIndex:0];
      }

      ushort featureChannels = X.featureChannels;
      if (channel_offset % 4 > 0) {
        featureChannels += tex_offset;
      }
      const auto& launchParams =
          metal::mpscnn::spatialPointwiseKernelLaunchParams(
              state, X.numberOfImages, featureChannels, X.height, X.width);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
    }

    channel_offset += X.featureChannels;

    {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer.buffer computeCommandEncoder];

      id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
          specializedPipelineState:"store_features"
                         Constants:@[
                           @(T.height),
                           @(T.width),
                           @(T.featureChannels),
                           @(T.numberOfImages),
                         ]];
      id<MTLBuffer> offsetBuffer = [[MetalContext sharedInstance].device
          newBufferWithLength:2 * sizeof(ushort)
                      options:MTLResourceCPUCacheModeWriteCombined];
      ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
      offsetBufferPtr[0] = channel_offset / 4;
      offsetBufferPtr[1] = (Y.featureChannels + 3) / 4;

      [encoder setComputePipelineState:state];
      [encoder setTexture:[Y texture] atIndex:0];
      [encoder setTexture:[T texture] atIndex:1];
      [encoder setBuffer:offsetBuffer offset:0 atIndex:0];

      const auto& launchParams =
          metal::mpscnn::spatialPointwiseKernelLaunchParams(state, T);
      [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
              threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
      [encoder endEncoding];
    }
  }
  auto output = makeTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat(const ITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(
      dim == 0 || dim == 1,
      "Metal cat is implemented only for batch dimension");
  int64_t cat_dim_size = 0;
  TORCH_CHECK(!tensors.empty(), "cat expected a non-empty list of Tensor");
  at::Tensor tensor = *tensors.begin();
  MetalCommandBuffer* commandBuffer = getCommandBuffer(tensor);
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dim() == 4, "Metal cat expects 4 dimensional inputs");
    TORCH_CHECK(t.is_metal(), "Metal cat expects metal tensors");

    for (int d = 0; d < 4; ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_CHECK(
          t.size(d) == tensor.size(d),
          "Metal cat inputs must have matching sizes except concatenated dimension");
    }
    cat_dim_size += t.size(dim);
  }
  auto result_size = tensor.sizes().vec();
  result_size[dim] = cat_dim_size;
  TORCH_CHECK(
      result_size[0] * ((result_size[1] + 3) / 4) > 1,
      "Output tensor must be a texture array");
  MetalTensorImplStorage mt{result_size};
  mt.texture()->setCommandBuffer(commandBuffer);
  mt.texture()->allocateTemporaryStorage(result_size, commandBuffer);

  if (dim == 1) {
    return cat_feature(tensor, tensors, mt);
  }
  return cat_batch(tensor, tensors, mt);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN(cat));
}

}
}
}
