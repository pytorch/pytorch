
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

Tensor cat_batch(const TensorList tensors, MetalTensorImplStorage& mt) {
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(tensor);
  MPSImage* Y = mt.texture()->image();
  ushort cat_dim4_pointer = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = getCommandBufferFromTensor(t);
    TORCH_CHECK(
        [commandBuffer isEqual:Xcb],
        @"inputs have different Metal command buffers");
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
        pipelineState:mpscnn::kernelFor(
                          X, @"copy_offset", @"copy_offset_nonarray")];
    id<MTLBuffer> offsetBuffer = [[MPSCNNContext sharedInstance].device
        newBufferWithLength:1 * sizeof(ushort)
                    options:MTLResourceOptionCPUCacheModeWriteCombined];
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
    [X markRead];
    cat_dim4_pointer += t.size(0) * ((t.size(1) + 3) / 4);
  }
  auto output = makeTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat_feature(const TensorList tensors, MetalTensorImplStorage& mt) {
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(tensor);
  MPSImage* Y = mt.texture()->image();
  ushort channel_offset = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = getCommandBufferFromTensor(t);
    TORCH_CHECK(
        [commandBuffer isEqual:Xcb],
        @"inputs have different Metal command buffers");
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    auto kernelString = metal::mpscnn::kernelFor(
        X, @"append_features_off0", @"append_features_off0_nonarray");
    ushort tex_offset = channel_offset % 4;
    if (tex_offset == 1) {
      kernelString = metal::mpscnn::kernelFor(
          X, @"append_features_off1", @"append_features_off1_nonarray");
    } else if (tex_offset == 2) {
      kernelString = metal::mpscnn::kernelFor(
          X, @"append_features_off2", @"append_features_off2_nonarray");
    } else if (tex_offset == 3) {
      kernelString = metal::mpscnn::kernelFor(
          X, @"append_features_off3", @"append_features_off3_nonarray");
    }

    id<MTLComputePipelineState> state =
        [[MPSCNNContext sharedInstance] pipelineState:kernelString];
    id<MTLBuffer> offsetBuffer = [[MPSCNNContext sharedInstance].device
        newBufferWithLength:5 * sizeof(ushort)
                    options:MTLResourceOptionCPUCacheModeWriteCombined];
    ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
    offsetBufferPtr[0] = (X.featureChannels + tex_offset + 3) / 4;
    offsetBufferPtr[1] = (Y.featureChannels + 3) / 4;
    offsetBufferPtr[2] = channel_offset / 4;
    offsetBufferPtr[3] = (X.featureChannels + 3) / 4;
    offsetBufferPtr[4] = X.numberOfImages * offsetBufferPtr[0];

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    [encoder setBuffer:offsetBuffer offset:0 atIndex:0];

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
    [X markRead];
    channel_offset += X.featureChannels;
  }
  auto output = makeTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat(const TensorList tensors, int64_t dim) {
  TORCH_CHECK(
      dim == 0 || dim == 1,
      "Metal cat is implemented only for batch dimension");
  int64_t cat_dim_size = 0;
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(tensor);
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
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
  mt.texture()->allocateTemporaryTextureStorage(result_size, commandBuffer);

  if (dim == 1) {
    return cat_feature(tensors, mt);
  }
  return cat_batch(tensors, mt);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("_cat", TORCH_FN(cat));
}

}
}
}
