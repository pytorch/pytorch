#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNFullyConnectedOp.h>
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
  auto weight_ =
      weight.t().view({weight.size(1), weight.size(0), 1, 1}).contiguous();
  // Reshape the input tensor to {N, C, 1, 1}
  auto input_ = input.view({input.size(0), input.size(1), 1, 1});
  MPSImage* X = imageFromTensor(input_);
  Conv2DParams params;
  params.N = X.numberOfImages;
  params.OC = weight_.size(0);
  params.IC = weight_.size(1);
  params.KH = params.KW = 1, params.H = params.W = 1;
  auto packedWeights =
      permuteWeights(weight_.data_ptr<float>(), weight_.sizes().vec());
  MetalTensorImplStorage mt{{params.N, params.OC}};
  SmallVector<int64_t, 4> textureSize = {params.N, params.OC, 1, 1};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input_);
  mt.texture()->allocateTemporaryStorage(textureSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  float* w = packedWeights.data();
  float* b = bias.data_ptr<float>();
  MPSCNNFullyConnectedOp* fc = [MPSCNNFullyConnectedOp linear:params
                                                      weights:w
                                                         bias:b
                                                 neuronFilter:NeuronType::None];
  [fc encode:commandBuffer.buffer sourceImage:X destinationImage:Y];
  // The output texture becomes {N, oC, 1, 1}. Reshape it to {N, oC}
  auto output =
      makeTensor(std::move(mt), input.options()).view({params.N, params.OC});
  return output;
}

namespace prepack {

Tensor linear(const Tensor& input, LinearOpContext& context) {
  TORCH_CHECK(input.is_metal());
  TORCH_CHECK(context.get_weight().device() == kCPU);
  TORCH_CHECK(context.get_weight().dim() == 4);
  // Reshape the input tensor to {N, C, 1, 1}
  auto input_ = input.view({input.size(0), input.size(1), 1, 1});
  MPSImage* X = imageFromTensor(input_);
  Conv2DParams params;
  params.N = X.numberOfImages;
  params.OC = context.get_weight().size(0);
  params.IC = context.get_weight().size(1);
  params.KH = params.KW = 1;
  params.H = params.W = 1;
  MPSCNNFullyConnectedOp* op =
      (__bridge MPSCNNFullyConnectedOp*)(context.get_opaqueOpPtr());
  NeuronType nt =
      neuronType(context.get_output_min(), context.get_output_max());
  if (!op) {
    float* w = context.get_weight().data_ptr<float>();
    float* b = context.get_bias().has_value()
        ? ((*context.get_bias()).data_ptr<float>())
        : nullptr;
    op = [MPSCNNFullyConnectedOp linear:params
                                weights:w
                                   bias:b
                           neuronFilter:nt];
    context.set_opaqueOpPtr((void*)CFBridgingRetain(op));
    context.set_releaseCallback(^(void* res) {
      if (res) {
        CFBridgingRelease(res);
      }
    });
  }
  MetalTensorImplStorage mt{{params.N, params.OC}};
  SmallVector<int64_t, 4> textureSize = {params.N, params.OC, 1, 1};
  MetalCommandBuffer* commandBuffer = getCommandBufferFromTensor(input_);
  mt.texture()->allocateTemporaryStorage(textureSize, commandBuffer);
  MPSImage* Y1 = mt.texture()->image();
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y1];
  if (nt == NeuronType::Clamp) {
    MPSImage* Y2 = createTemporaryImage(commandBuffer, [Y1 sizes]);
    float min = context.get_output_min().value().toFloat();
    float max = context.get_output_max().value().toFloat();
    MPSCNNClampOp* clampOp =
        [MPSCNNClampOp newWithTextures:@[ Y1, Y2 ] Args:@[ @(min), @(max) ]];
    [clampOp encode:commandBuffer.buffer];
    mt.texture()->setImage(Y2);
  }
  // The output texture becomes {N, oC, 1, 1}. Reshape it to {N, oC}
  auto output =
      makeTensor(std::move(mt), input.options()).view({params.N, params.OC});
  return output;
}

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context) {
  return linear(input, *op_context);
}

}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("addmm", TORCH_FN(addmm));
};

TORCH_LIBRARY_IMPL(metal_prepack, Metal, m) {
  m.impl("linear_run", TORCH_FN(prepack::linear_run));
}

}
}
}
