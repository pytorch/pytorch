#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

template <typename T>
Tensor mpscnn_softmax(
    const Tensor& input,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(input.is_metal());
  // TODO: [T87180544] Implement softmax/log_softmax in metal shaders
  TORCH_CHECK(input.dim() == 2);
  if(input.numel() == 0){
      return makeTensor({input.sizes().vec()}, input.options());
  }
  std::vector<int64_t> newSize(4, 1);
  if (dim == 0) {
    newSize[1] = input.size(0);
    newSize[2] = input.size(1);
  } else {
    newSize[0] = input.size(0);
    newSize[1] = input.size(1);
  }
  auto input_ = input.view(newSize);
  MPSImage* X = imageFromTensor(input_);
  // MPSCNNSoftmax kernels operate on feature channels
  // https://developer.apple.com/documentation/metalperformanceshaders/mpscnnsoftmax?changes=_1&language=objc
  T* softmax = [[T alloc] initWithDevice:[MetalContext sharedInstance].device];
  MetalTensorImplStorage mt{newSize};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input_);
  mt.texture()->allocateTemporaryStorage(newSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [softmax encodeToCommandBuffer:commandBuffer.buffer
                     sourceImage:X
                destinationImage:Y];
  // restore the original sizes
  auto output = makeTensor(std::move(mt), input.options()).view(input.sizes());
  return output;
}

Tensor log_softmax_int(
    const Tensor& input,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return mpscnn_softmax<MPSCNNLogSoftMax>(input, dim, dtype);
}

Tensor softmax_int(
    const Tensor& input,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return mpscnn_softmax<MPSCNNSoftMax>(input, dim, dtype);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::log_softmax.int"), TORCH_FN(metal::log_softmax_int));
  m.impl(TORCH_SELECTIVE_NAME("aten::softmax.int"), TORCH_FN(metal::softmax_int));
};

}
}
}
