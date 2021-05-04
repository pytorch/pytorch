#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(weight.device() == kCPU && weight.dim() == 2);
  TORCH_CHECK(bias.device() == kCPU);
  TORCH_CHECK(beta.toFloat() == 1.0f);
  TORCH_CHECK(alpha.toFloat() == 1.0f);
  // Here we treat the matrix multiplication as convolution
  auto weight_ = weight.t()
                     .view({weight.sizes()[1], weight.sizes()[0], 1, 1})
                     .contiguous();
  // Permute the input texture to become {N, C, 1, 1}
  auto input_ = input.view({input.sizes()[0], input.sizes()[1], 1, 1});
  MPSImage* X = imageFromTensor(input_);
  const int64_t N = X.numberOfImages;
  const int64_t oC = weight_.sizes()[0];
  const int64_t kH = X.height;
  const int64_t kW = X.width;
  const int64_t iC = weight_.sizes()[1] / kH / kW;
  auto packedWeights =
      permuteWeights(weight_.data_ptr<float>(), {oC, iC, kH, kW});
  MPSCNNConvolutionDescriptor* desc =
      [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kW
                                                              kernelHeight:kH
                                                      inputFeatureChannels:iC
                                                     outputFeatureChannels:oC
                                                              neuronFilter:nil];
  desc.strideInPixelsX = 1;
  desc.strideInPixelsY = 1;
  MPSCNNConvDataSource* ds = [[MPSCNNConvDataSource alloc]
      initWithWeights:packedWeights.data()
                 Bias:bias.defined() ? bias.data_ptr<float>() : nil
                 Desc:desc];
  MPSCNNFullyConnected* fc = nil;
  if (@available(iOS 11.0, *)) {
    fc = [[MPSCNNFullyConnected alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
               weights:ds];
  } else {
#if TARGET_OS_IPHONE
    fc = [[MPSCNNFullyConnected alloc]
               initWithDevice:[MPSCNNContext sharedInstance].device
        convolutionDescriptor:desc
                kernelWeights:(float*)packedWeights.data()
                    biasTerms:bias.defined() ? bias.data_ptr<float>() : nil
                        flags:MPSCNNConvolutionFlagsNone];
#endif
  }
  [fc setClipRect:MTLRegionMake3D(0, 0, 0, 1, 1, N)];
  [fc setOffset:{.x = static_cast<NSInteger>(X.width / 2),
                 .y = static_cast<NSInteger>(X.height / 2),
                 .z = 0}];
  std::vector<int64_t> textureSize = {N, oC, 1, 1};
  MetalTensorImplStorage mt{{N, oC}};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(textureSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [fc encodeToCommandBuffer:commandBuffer.buffer
                sourceImage:X
           destinationImage:Y];
  // The output texture becomes {N, oC, 1, 1}. Make it {1, 1, N, oC}
  auto output = makeTensor(std::move(mt), input.options()).view({N, oC});
  return output;
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("addmm", TORCH_FN(addmm));
};

}
}
}
