#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;

Tensor& hardswish_(Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  IntArrayRef outputSize = input.sizes();
  std::vector<int64_t> imageSize = computeImageSize(outputSize);
  MPSImage* Y = createTemporaryImage(commandBuffer, imageSize);
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:mpscnn::kernelFor(
                                   X, "hardswish", "hardswish_nonarray")
                     Constants:@[
                       @(X.featureChannels),
                       @(X.height),
                       @(X.width)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X markRead];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->setImage(Y);
  return input;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("hardswish_", TORCH_FN(hardswish_));
};

}
}
}
