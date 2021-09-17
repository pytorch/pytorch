#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;

Tensor neuronKernel(const Tensor& input, MPSCNNNeuron* neuron) {
  MPSImage* X = imageFromTensor(input);
  IntArrayRef outputSize = input.sizes();
  if(input.numel() == 0){
    return makeTensor({outputSize.vec()}, input.options());
  }
  IntArrayRef textureSize = outputSize;
  MetalTensorImplStorage mt{outputSize.vec()};
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  mt.texture()->allocateTemporaryStorage(textureSize, commandBuffer);
  MPSImage* Y = mt.texture()->image();
  [neuron encodeToCommandBuffer:commandBuffer.buffer
                    sourceImage:X
               destinationImage:Y];
  auto output = makeTensor(std::move(mt), input.options());
  return output;
}

Tensor& neuronKernel_(Tensor& input, MPSCNNNeuron* neuron) {
  MPSImage* X = imageFromTensor(input);
  IntArrayRef outputSize = input.sizes();
  if(input.numel() == 0){
    return input;
  }
  IntArrayRef textureSize = outputSize;
  MetalCommandBuffer* commandBuffer = getCommandBuffer(input);
  MPSImage* Y = createTemporaryImage(commandBuffer, textureSize);
  [neuron encodeToCommandBuffer:commandBuffer.buffer
                    sourceImage:X
               destinationImage:Y];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->setImage(Y);
  return input;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor relu(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return neuronKernel(input, [MPSCNNNeuronOp relu]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& relu_(Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return neuronKernel_(input, [MPSCNNNeuronOp relu]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor sigmoid(const Tensor& input) {
  return neuronKernel(input, [MPSCNNNeuronOp sigmoid]);
}

API_AVAILABLE(ios(11.0), macos(10.13))
Tensor& hardsigmoid_(Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return neuronKernel_(input, [MPSCNNNeuronOp hardSigmoid]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor tanh(const Tensor& input) {
  TORCH_CHECK(input.is_metal());
  return neuronKernel(input, [MPSCNNNeuronOp tanh]);
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("tanh", tanh);
  m.impl("relu", TORCH_FN(relu));
  m.impl("relu_", TORCH_FN(relu_));
  m.impl("sigmoid", TORCH_FN(sigmoid));
  if (@available(iOS 11.0, *)) {
    m.impl("hardsigmoid_", TORCH_FN(hardsigmoid_));
  }
};

}
}
}
