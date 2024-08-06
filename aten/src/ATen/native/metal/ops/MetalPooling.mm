#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at::native::metal {

API_AVAILABLE(ios(11.0), macos(10.13))
static Tensor max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(input.dim() == 3 || input.dim() == 4);
  TORCH_CHECK(
      dilation[0] == dilation[1] == 1, "dilation is not supported on MPSCNN");
  const int64_t iN = input.sizes()[0];
  const int64_t iC = input.sizes()[1];
  const int64_t iH = input.sizes()[2];
  const int64_t iW = input.sizes()[3];
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size[1];
  const int64_t sH = stride[0];
  const int64_t sW = stride[1];
  const int64_t pH = padding[0];
  const int64_t pW = padding[1];
  const int64_t dH = dilation[0];
  const int64_t dW = dilation[1];
  int64_t oN = iN;
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  SmallVector<int64_t, 4>outputSize{oN, oC, oH, oW};
  if(input.numel() == 0){
    return makeTensor({IntArrayRef(outputSize).vec()}, input.options());
  }
  MPSImage* X = imageFromTensor(input);
  MPSCNNPoolingMax* pool = [[MPSCNNPoolingMax alloc]
       initWithDevice:[MetalContext sharedInstance].device
          kernelWidth:kernel_size[0]
         kernelHeight:kernel_size[1]
      strideInPixelsX:stride[0]
      strideInPixelsY:stride[1]];
  [pool setEdgeMode:MPSImageEdgeModeClamp];
  [pool
      setOffset:{.x = mpscnn::computeMPSAlignOffset(kernel_size[0], padding[0]),
                 .y = mpscnn::computeMPSAlignOffset(kernel_size[1], padding[1]),
                 .z = 0}];
  MetalTensorImplStorage mt{IntArrayRef(outputSize).vec()};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(outputSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [pool encodeToCommandBuffer:commandBuffer.buffer
                  sourceImage:X
             destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(11.0), macos(10.13))
static Tensor adaptive_avg_pool2d(const Tensor& input, IntArrayRef output_size) {
  // averages across the width and height, and outputs a 1x1xC image.
  TORCH_CHECK(output_size[0] == 1 && output_size[1] == 1);
  TORCH_CHECK(input.is_metal());
  SmallVector<int64_t, 4> outputSize{
      input.sizes()[0], input.sizes()[1], output_size[0], output_size[1]};
  if(input.numel() == 0){
      return makeTensor({IntArrayRef(outputSize).vec()}, input.options());
  }
  MPSImage* X = imageFromTensor(input);
  MPSCNNPoolingAverage* pool = [[MPSCNNPoolingAverage alloc]
       initWithDevice:[MetalContext sharedInstance].device
          kernelWidth:X.width
         kernelHeight:X.height
      strideInPixelsX:X.width
      strideInPixelsY:X.height];
  [pool setEdgeMode:MPSImageEdgeModeClamp];
  [pool setOffset:{.x = static_cast<NSInteger>(X.width / 2),
                   .y = static_cast<NSInteger>(X.height / 2),
                   .z = 0}];

  MetalTensorImplStorage mt{IntArrayRef(outputSize).vec()};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(outputSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [pool encodeToCommandBuffer:commandBuffer.buffer
                  sourceImage:X
             destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::max_pool2d"), TORCH_FN(max_pool2d));
  m.impl(TORCH_SELECTIVE_NAME("aten::adaptive_avg_pool2d"), TORCH_FN(adaptive_avg_pool2d));
}

} // namespace at::native::metal
