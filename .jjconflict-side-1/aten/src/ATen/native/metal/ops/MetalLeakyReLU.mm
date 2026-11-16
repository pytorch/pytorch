#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#include <torch/library.h>

namespace at::native::metal {

using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;

static Tensor& leaky_relu_(Tensor& input, const Scalar& negative_slope_val) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  IntArrayRef outputSize = input.sizes();
  std::vector<int64_t> imageSize = computeImageSize(outputSize);
  float negative_slope = negative_slope_val.toFloat();
  MPSImage* Y = createTemporaryImage(commandBuffer, imageSize);
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state =
      [[MetalContext sharedInstance] specializedPipelineState:"leaky_relu"
                                                    Constants:@[
                                                      @(X.numberOfImages),
                                                      @(X.featureChannels),
                                                      @(X.height),
                                                      @(X.width),
                                                      @(negative_slope)
                                                    ]];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->setImage(Y);
  return input;
}

static Tensor leaky_relu(const at::Tensor& input, const Scalar& negative_slope_val) {
  MPSImage* X = imageFromTensor(input);
  IntArrayRef outputSize = input.sizes();
  MetalTensorImplStorage mt{outputSize.vec()};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(outputSize, commandBuffer);
  float negative_slope = negative_slope_val.toFloat();
  MPSImage* Y = mt.texture()->image();
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state =
      [[MetalContext sharedInstance] specializedPipelineState:"leaky_relu"
                                                    Constants:@[
                                                      @(X.numberOfImages),
                                                      @(X.featureChannels),
                                                      @(X.height),
                                                      @(X.width),
                                                      @(negative_slope)
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

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu_"), TORCH_FN(leaky_relu_));
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu"), TORCH_FN(leaky_relu));
}

} // namespace at::native::metal
