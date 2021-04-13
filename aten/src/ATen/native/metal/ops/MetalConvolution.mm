#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#import <ATen/native/metal/ops/MetalConvolution.h>

#import <ATen/ATen.h>

namespace at {
namespace native {
namespace metal {

using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(input.is_metal());
  Conv2DParams params{
      input.sizes(), weight.sizes(), padding, stride, dilation, groups};
  TORCH_INTERNAL_ASSERT(input.dim() == 4, "Expected 4-dimensional input");
  TORCH_INTERNAL_ASSERT(weight.dim() == 4, "Expected 4-dimensional weight");
  TORCH_CHECK(weight.device().type() == kCPU);
  MPSImage* X = imageFromTensor(input);
  const int64_t oC = weight.sizes()[0];
  const int64_t iC = weight.sizes()[1];
  const int64_t kH = weight.sizes()[2];
  const int64_t kW = weight.sizes()[3];
  auto packedWeights = at::native::metal::permuteWeights(
      weight.data_ptr<float>(), {oC, iC, kH, kW});
  // MPSCNN Convolution
  float* w = packedWeights.data();
  float* b = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  MPSCNNConvOp* op = [MPSCNNConvOp conv2d:params
                                  weights:w
                                     bias:b
                             neuronFilter:NeuronType::None];
  auto outputSize = params.output_sizes();
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

namespace prepack {

Tensor conv2d(const Tensor& input, Conv2dOpContext& context) {
  MPSImage* X = imageFromTensor(input);
  Conv2DParams params{input.sizes(),
                      context.weight.sizes(),
                      context.padding,
                      context.stride,
                      context.dilation,
                      context.groups};
  MPSCNNConvOp* op = (__bridge MPSCNNConvOp*)(context.conv2dOp);
  NeuronType nt = neuronType(context);
  if (!op) {
    float* w = context.weight.data_ptr<float>();
    float* b = context.bias.has_value() ? ((*context.bias).data_ptr<float>())
                                        : nullptr;
    op = [MPSCNNConvOp conv2d:params weights:w bias:b neuronFilter:nt];
    context.conv2dOp = (void*)CFBridgingRetain(op);
    context.releaseCallback = ^(void* res) {
      if (res) {
        CFBridgingRelease(res);
      }
    };
  }

  auto outputSize = params.output_sizes();
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y1 = mt.texture()->image();
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y1];
  // fuse hardtanh with convolution
  if (nt == NeuronType::Clamp) {
    MPSImage* Y2 = createTemporaryImage(commandBuffer, [Y1 sizes]);
    float min = context.output_min.value().toFloat();
    float max = context.output_max.value().toFloat();
    MPSCNNClampOp* clampOp =
        [MPSCNNClampOp newWithTextures:@[ Y1, Y2 ] Args:@[ @(min), @(max) ]];
    [clampOp encode:commandBuffer.buffer];
    mt.texture()->copyFromTexture(Y2);
  }
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

} // namespace prepack

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("conv2d", TORCH_FN(conv2d));
};

}
}
}
