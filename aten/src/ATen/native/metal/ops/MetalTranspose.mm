#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

Tensor t(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.dim() == 2);
  auto strides = input.strides().vec();
  auto sizes = input.sizes().vec();
  MPSImage* X = imageFromTensor(input);
  TORCH_CHECK(X.numberOfImages == 1);
  TORCH_CHECK(X.featureChannels == 1);
  MetalTensorImplStorage mt({sizes[1], sizes[0]});
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(
      {1, 1, sizes[1], sizes[0]}, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  MPSImageTranspose* transpose = [[MPSImageTranspose alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];
  [transpose encodeToCommandBuffer:commandBuffer.buffer
                       sourceImage:X
                  destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

}
}
}
