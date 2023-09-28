#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorUtils.h>
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
  auto outputSize = params.output_sizes();
  if(c10::multiply_integers(outputSize) == 0){
      return makeTensor({outputSize}, input.options());
  }
  MPSImage* X = imageFromTensor(input);
  auto packedWeights = weight.contiguous(c10::MemoryFormat::ChannelsLast);
  // MPSCNN Convolution
  float* w = packedWeights.data_ptr<float>();
  float* b = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  MPSCNNConvOp* op = [MPSCNNConvOp conv2d:params
                                  weights:w
                                     bias:b
                             neuronFilter:NeuronType::None];
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(outputSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

namespace prepack {

Tensor conv2d(const Tensor& input, Conv2dOpContext& context) {
  MPSImage* X = imageFromTensor(input);
  Conv2DParams params{input.sizes(),
                      context.get_weight().sizes(),
                      context.get_padding(),
                      context.get_stride(),
                      context.get_dilation(),
                      context.get_groups()};
  auto outputSize = params.output_sizes();
  if(c10::multiply_integers(outputSize) == 0){
    return makeTensor({outputSize}, input.options());
  }
  MPSCNNConvOp* op = (__bridge MPSCNNConvOp*)(context.get_conv2dOpPtr());
  NeuronType nt = neuronType(context.get_output_min(), context.get_output_max());
  if (!op) {
    float* w = context.get_weight().data_ptr<float>();
    float* b = context.get_bias().has_value() ? ((*context.get_bias()).data_ptr<float>())
                                        : nullptr;
    op = [MPSCNNConvOp conv2d:params weights:w bias:b neuronFilter:nt];
    context.set_conv2dOpPtr((void*)CFBridgingRetain(op));
    context.set_releaseCallback(^(void* res) {
      if (res) {
        CFBridgingRelease(res);
      }
    });
  }
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(outputSize, commandBuffer);
  MPSImage* Y1 = mt.texture()->image();
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y1];
  // fuse hardtanh with convolution
  if (nt == NeuronType::Clamp) {
    MPSImage* Y2 = createTemporaryImage(commandBuffer, [Y1 sizes]);
    float min = context.get_output_min().value().toFloat();
    float max = context.get_output_max().value().toFloat();
    MPSCNNClampOp* clampOp =
        [MPSCNNClampOp newWithTextures:@[ Y1, Y2 ] Args:@[ @(min), @(max) ]];
    [clampOp encode:commandBuffer.buffer];
    mt.texture()->setImage(Y2);
  }
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

static Tensor conv2d_prepack_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& op_context) {
  return conv2d(input, *op_context);
}

} // namespace prepack

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  // NB: this didn't actually do anything; need to generalize this to
  // work for general convolution and register to aten::convolution
  // m.impl(TORCH_SELECTIVE_NAME("aten::conv2d"), TORCH_FN(conv2d));
};

TORCH_LIBRARY_IMPL(metal_prepack, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::conv2d_run"), prepack::conv2d_prepack_run);
}

}
}
}
