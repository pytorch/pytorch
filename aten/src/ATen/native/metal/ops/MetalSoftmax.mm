#include <ATen/Tensor.h>
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

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor log_softmax_int(
    const Tensor& input,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(dim == 1);
  TORCH_CHECK(input.is_metal());
  MPSImage* X = imageFromTensor(input);
  TORCH_CHECK(X.height == 1 && X.width == 1);
  std::vector<int64_t> outputSize = input.sizes().vec();
  MPSCNNLogSoftMax* logSoftmax = [[MPSCNNLogSoftMax alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];

  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(
      {outputSize[0], outputSize[1], 1, 1}, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [logSoftmax encodeToCommandBuffer:commandBuffer.buffer
                        sourceImage:X
                   destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("log_softmax.int", TORCH_FN(log_softmax_int));
};

}
}
}
