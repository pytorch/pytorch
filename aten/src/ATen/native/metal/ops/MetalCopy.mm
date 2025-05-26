#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <torch/library.h>

namespace at::native::metal {

static Tensor copy_to_host(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  MPSImage* X = imageFromTensor(input);
  if (X && !X.isTemporaryImage) {
    return input;
  }
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  auto&& sizes = [X sizes];
  MetalTensorImplStorage mt{sizes};
  mt.texture()->setCommandBuffer(commandBuffer);
  mt.texture()->allocateStorage(sizes);
  MPSImage* Y = mt.texture()->image();

  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   X, "copy", "copy_nonarray")
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
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

TORCH_LIBRARY_IMPL(metal, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("metal::copy_to_host"), TORCH_FN(copy_to_host));
}

} // namespace at::native::metal
