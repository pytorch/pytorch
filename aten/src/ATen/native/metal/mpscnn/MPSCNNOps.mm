#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensor.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNN.h>
#import <ATen/native/metal/mpscnn/MPSCNNClampOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNConvOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNNeuronOp.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageWrapper.h>

#include <ATen/InferSize.h>
#include <ATen/native/Pool.h>
#include <ATen/native/UpSample.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace metal {
namespace mpscnn {

using MetalTensor = at::native::metal::MetalTensor;
using MetalTensorImpl = at::MetalTensorImpl<MetalTensor>;

API_AVAILABLE(ios(10.0), macos(10.13))
static inline MPSImage* imageFromMetalTensor(const MetalTensor& tensor) {
  return tensor.texture()->image();
}

API_AVAILABLE(ios(10.0), macos(10.13))
static inline MPSImage* imageFromTensor(const Tensor& tensor) {
  TORCH_CHECK(tensor.is_metal());
  MetalTensorImpl* impl = (MetalTensorImpl*)tensor.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  return imageFromMetalTensor(metalTensor);
}

API_AVAILABLE(ios(10.0), macos(10.13))
static inline MetalCommandBuffer* commandBufferFromInputTensor(
    const Tensor& tensor) {
  TORCH_CHECK(tensor.is_metal());
  MetalTensorImpl* impl = (MetalTensorImpl*)tensor.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  MetalCommandBuffer* cmdBuffer = metalTensor.texture()->commandBuffer();
  TORCH_CHECK(cmdBuffer, @"Command Buffer can't be nil!");
  return cmdBuffer;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const Conv2DParams& params,
    NeuronType t) {
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
                             neuronFilter:t];
  auto outputSize = params.output_sizes();
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
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
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y1 = imageFromMetalTensor(mt);
  [op encode:commandBuffer.buffer sourceImage:X destinationImage:Y1];
  // fuse hardtanh with convolution
  if (nt == NeuronType::Clamp) {
    MPSImage* Y2 = [MPSImage temporaryImageFromSize:[Y1 sizes]
                                      commandBuffer:commandBuffer];
    float min = context.output_min.value().toFloat();
    float max = context.output_max.value().toFloat();
    MPSCNNClampOp* clampOp =
        [MPSCNNClampOp newWithTextures:@[ Y1, Y2 ] Args:@[ @(min), @(max) ]];
    [clampOp encode:commandBuffer.buffer];
    mt.texture()->copyFromTexture(Y2);
  }
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
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
  MPSImage* X = imageFromTensor(input);
  MPSCNNPoolingMax* pool = [[MPSCNNPoolingMax alloc]
       initWithDevice:[MPSCNNContext sharedInstance].device
          kernelWidth:kernel_size[0]
         kernelHeight:kernel_size[1]
      strideInPixelsX:stride[0]
      strideInPixelsY:stride[1]];
  [pool setEdgeMode:MPSImageEdgeModeClamp];
  [pool setOffset:{.x = computeMPSAlignOffset(kernel_size[0], padding[0]),
                   .y = computeMPSAlignOffset(kernel_size[1], padding[1]),
                   .z = 0}];
  int64_t oN = iN;
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);

  std::vector<int64_t> outputSize{oN, oC, oH, oW};
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [pool encodeToCommandBuffer:commandBuffer.buffer
                  sourceImage:X
             destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor global_avg_pool2d(const Tensor& input, IntArrayRef output_size) {
  MPSImage* X = imageFromTensor(input);
  MPSCNNPoolingAverage* pool = [[MPSCNNPoolingAverage alloc]
       initWithDevice:[MPSCNNContext sharedInstance].device
          kernelWidth:X.width
         kernelHeight:X.height
      strideInPixelsX:X.width
      strideInPixelsY:X.height];
  [pool setEdgeMode:MPSImageEdgeModeClamp];
  [pool setOffset:{.x = static_cast<NSInteger>(X.width / 2),
                   .y = static_cast<NSInteger>(X.height / 2),
                   .z = 0}];
  std::vector<int64_t> outputSize{
      input.sizes()[0], input.sizes()[1], output_size[0], output_size[1]};
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [pool encodeToCommandBuffer:commandBuffer.buffer
                  sourceImage:X
             destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor neuronKernel(const Tensor& input, MPSCNNNeuron* neuron) {
  MPSImage* X = imageFromTensor(input);
  std::vector<int64_t> outputSize = input.sizes().vec();
  std::vector<int64_t> textureSize = outputSize;
  if (input.dim() == 2) {
    textureSize = {outputSize[0], outputSize[1], 1, 1};
  }
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(textureSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [neuron encodeToCommandBuffer:commandBuffer.buffer
                    sourceImage:X
               destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& neuronKernel_(Tensor& input, MPSCNNNeuron* neuron) {
  MPSImage* X = imageFromTensor(input);
  std::vector<int64_t> outputSize = input.sizes().vec();
  std::vector<int64_t> textureSize = outputSize;
  if (input.dim() == 2) {
    textureSize = {outputSize[0], outputSize[1], 1, 1};
  }
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  MPSImage* Y = [MPSImage temporaryImageFromSize:input.sizes().vec()
                                   commandBuffer:commandBuffer];
  [neuron encodeToCommandBuffer:commandBuffer.buffer
                    sourceImage:X
               destinationImage:Y];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor relu(const Tensor& input) {
  return neuronKernel(input, [MPSCNNNeuronOp relu]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& relu_(Tensor& input) {
  return neuronKernel_(input, [MPSCNNNeuronOp relu]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor sigmoid(const Tensor& input) {
  return neuronKernel(input, [MPSCNNNeuronOp sigmoid]);
}

API_AVAILABLE(ios(11.0), macos(10.13))
Tensor& hardsigmoid_(Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  std::vector<int64_t> outputSize = input.sizes().vec();
  std::vector<int64_t> textureSize = outputSize;
  if (input.dim() == 2) {
    textureSize = {outputSize[0], outputSize[1], 1, 1};
  }
  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(textureSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  static dispatch_once_t onceToken;
  static MPSCNNNeuronHardSigmoid* neuron = nil;
  dispatch_once(&onceToken, ^{
    neuron = [[MPSCNNNeuronHardSigmoid alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
                     a:1.0/6.0
                     b:0.5];
  });
  [neuron encodeToCommandBuffer:commandBuffer.buffer
                    sourceImage:X
               destinationImage:Y];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor tanh(const Tensor& input) {
  return neuronKernel(input, [MPSCNNNeuronOp tanh]);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& hardtanh_(Tensor& input, Scalar min_val, Scalar max_val) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  MPSImage* Y = [MPSImage temporaryImageFromSize:input.sizes().vec()
                                   commandBuffer:commandBuffer];
  float min = min_val.toFloat();
  float max = max_val.toFloat();
  MPSCNNClampOp* clampOp = [MPSCNNClampOp newWithTextures:@[ X, Y ]
                                                     Args:@[ @(min), @(max) ]];
  [clampOp encode:commandBuffer.buffer];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input;
}

Tensor& hardswish_(Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  std::vector<int64_t> outputSize = input.sizes().vec();
  std::vector<int64_t> textureSize = outputSize;
  if (input.dim() == 2) {
    textureSize = {outputSize[0], outputSize[1], 1, 1};
  }
  MPSImage* Y = [MPSImage temporaryImageFromSize:textureSize commandBuffer:commandBuffer];
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   X, @"hardswish", @"hardswish_nonarray")
                     Constants:@[
                       @(X.featureChannels),
                       @(X.height),
                       @(X.width)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X markRead];
  MetalTensorImpl* impl = (MetalTensorImpl*)input.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input;
}

/*
 A fully connected layer takes an MPSImage object with dimensions source.width x
 source.height x Ni, convolves it with
 Weights[No][source.width][source.height][Ni],and produces a 1 x 1 x No output.

 Thus, the following conditions must be true:
 kernelWidth == source.width
 kernelHeight == source.height
 clipRect.size.width == 1
 clipRect.size.height == 1

 You can think of a fully connected layer as a matrix multiplication
 where the image is flattened into a vector of length
 source.width*source.height*Ni, and the weights are arranged in a matrix of
 dimension No x (source.width*source.height*Ni) to produce an output vector of
 length No

 The value of the strideInPixelsX, strideInPixelsY, and groups properties must
 be 1. The offset property is not applicable and it is ignored. Because the clip
 rectangle is clamped to the destination image bounds, if the destination is 1 x
 1, you do not need to set the clipRect property.
 */
API_AVAILABLE(ios(10.0), macos(10.13))
Tensor addmm(const Tensor& bias, const Tensor& input, const Tensor& weight) {
  MPSImage* X = imageFromTensor(input);
  const int64_t N = X.numberOfImages;
  const int64_t oC = weight.sizes()[0];
  const int64_t kH = X.height;
  const int64_t kW = X.width;
  const int64_t iC = weight.sizes()[1] / kH / kW;
  auto packedWeights = at::native::metal::permuteWeights(
      weight.data_ptr<float>(), {oC, iC, kH, kW});
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
  std::vector<int64_t> outputSize = {N, oC, 1, 1};
  MetalTensor mt{{N, oC}};

  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSize, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [fc encodeToCommandBuffer:commandBuffer.buffer
                sourceImage:X
           destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

bool broadCastFirstInput(const Tensor& input1, const Tensor& input2) {
  if (
    (input2.sizes()[2] > 1 && input1.sizes()[2] == 1) ||
    (input2.sizes()[3] > 1 && input1.sizes()[3] == 1)
  ) {
    return true;
  }
  return false;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor binaryElementwiseShaderKernel(
    const Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(input1, input2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensor mt{outputSize};
  MetalCommandBuffer* cb1 = commandBufferFromInputTensor(input1);
  MetalCommandBuffer* cb2 = commandBufferFromInputTensor(input2);
  TORCH_CHECK([cb1 isEqual:cb2], @"inputs have different command buffer");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = imageFromMetalTensor(mt);
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:kernelFor(X1, arrayKernel, nonarrayKernel)];
  id<MTLComputeCommandEncoder> encoder = [cb1.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X1 texture] atIndex:0];
  [encoder setTexture:[X2 texture] atIndex:1];
  [encoder setTexture:[Y texture] atIndex:2];
  const auto& launchParams = spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X1 markRead];
  [X2 markRead];
  auto output = MetalTensor::toTensor(std::move(mt), input1.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& binaryElementwiseShaderKernel_(
    Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(input1, input2)) {
    outputSize = input2.sizes().vec();
  }
  MetalCommandBuffer* cb1 = commandBufferFromInputTensor(input1);
  MetalCommandBuffer* cb2 = commandBufferFromInputTensor(input2);
  TORCH_CHECK([cb1 isEqual:cb2], @"inputs have different command buffer");
  MPSImage* Y = [MPSImage temporaryImageFromSize:outputSize commandBuffer:cb1];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      pipelineState:kernelFor(X1, arrayKernel, nonarrayKernel)];
  id<MTLComputeCommandEncoder> encoder = [cb1.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X1 texture] atIndex:0];
  [encoder setTexture:[X2 texture] atIndex:1];
  [encoder setTexture:[Y texture] atIndex:2];
  const auto& launchParams = spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X1 markRead];
  [X2 markRead];
  MetalTensorImpl* impl = (MetalTensorImpl*)input1.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input1;
}

template <typename T>
API_AVAILABLE(ios(11.3), macos(10.13))
Tensor binaryElementwiseMPSCNNKernel(
    const Tensor& input1,
    const Tensor& input2) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(input1, input2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensor mt{outputSize};
  MetalCommandBuffer* cb1 = commandBufferFromInputTensor(input1);
  MetalCommandBuffer* cb2 = commandBufferFromInputTensor(input2);
  TORCH_CHECK([cb1 isEqual:cb2], @"inputs have different command buffer");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = imageFromMetalTensor(mt);
  T* kernel = [[T alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];
  kernel.primaryStrideInPixelsY = (NSUInteger)(input1.sizes()[2] == 1 ? 0 : 1);
  kernel.primaryStrideInPixelsX = (NSUInteger)(input1.sizes()[3] == 1 ? 0 : 1);
  kernel.secondaryStrideInPixelsY = (NSUInteger)(input2.sizes()[2] == 1 ? 0 : 1);
  kernel.secondaryStrideInPixelsX = (NSUInteger)(input2.sizes()[3] == 1 ? 0 : 1);
  [kernel encodeToCommandBuffer:cb1.buffer
      primaryImage:X1
      secondaryImage:X2
      destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input1.options());
  return output;
}

template <typename T>
API_AVAILABLE(ios(11.3), macos(10.13))
Tensor& binaryElementwiseMPSCNNKernel_(
    Tensor& input1,
    const Tensor& input2) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(input1, input2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensor mt{outputSize};
  MetalCommandBuffer* cb1 = commandBufferFromInputTensor(input1);
  MetalCommandBuffer* cb2 = commandBufferFromInputTensor(input2);
  TORCH_CHECK([cb1 isEqual:cb2], @"inputs have different command buffer");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = imageFromMetalTensor(mt);
  T* kernel = [[T alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];
  [kernel encodeToCommandBuffer:cb1.buffer
      primaryImage:X1
      secondaryImage:X2
      destinationImage:Y];
  MetalTensorImpl* impl = (MetalTensorImpl*)input1.unsafeGetTensorImpl();
  MetalTensor& metalTensor = impl->unsafe_opaque_handle();
  metalTensor.texture()->copyFromTexture(Y);
  return input1;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor add(const Tensor& input1, const Tensor& input2) {
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNAdd>(input1, input2);
  }
  return binaryElementwiseShaderKernel(
      input1, input2, @"elementwise_add", @"elementwise_add_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& add_(Tensor& input1, const Tensor& input2) {
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNAdd>(input1, input2);
  }
  return binaryElementwiseShaderKernel_(
      input1, input2, @"elementwise_add", @"elementwise_add_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor sub(const Tensor& input1, const Tensor& input2) {
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNSubtract>(input1, input2);
  }
  return binaryElementwiseShaderKernel(
      input1, input2, @"elementwise_sub", @"elementwise_sub_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor mul(const Tensor& input1, const Tensor& input2) {
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNMultiply>(input1, input2);
  }
  return binaryElementwiseShaderKernel(
      input1, input2, @"elementwise_mul", @"elementwise_mul_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor t(const Tensor& input) {
  auto strides = input.strides().vec();
  auto sizes = input.sizes().vec();
  MPSImage* X = imageFromTensor(input);
  TORCH_CHECK(X.numberOfImages == 1);
  TORCH_CHECK(X.featureChannels == 1);
  MetalTensor mt({sizes[1], sizes[0]});
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(
      {1, 1, sizes[1], sizes[0]}, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  MPSImageTranspose* transpose = [[MPSImageTranspose alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];
  [transpose encodeToCommandBuffer:commandBuffer.buffer
                       sourceImage:X
                  destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor view(const Tensor& input, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, input.numel());
  auto stride =
      at::detail::computeStride(input.sizes(), input.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;

  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  MetalTensor mt{inferred_size, stride_value};
  mt.texture()->setCommandBuffer(commandBuffer);
  mt.texture()->copyFromTexture(X);
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

Tensor reshape(const Tensor& input, IntArrayRef shape) {
  return view(input, shape);
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor log_softmax_int(const Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  TORCH_CHECK(X.height == 1 && X.width == 1);
  std::vector<int64_t> outputSize = input.sizes().vec();
  MPSCNNLogSoftMax* logSoftmax = [[MPSCNNLogSoftMax alloc]
      initWithDevice:[MPSCNNContext sharedInstance].device];

  MetalTensor mt{outputSize};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(
      {outputSize[0], outputSize[1], 1, 1}, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  [logSoftmax encodeToCommandBuffer:commandBuffer.buffer
                        sourceImage:X
                   destinationImage:Y];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor upsample_nearest2d_vec(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize =
      upsample::compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = upsample::get_scale_value(scale_factors, 0);
  auto scale_w = upsample::get_scale_value(scale_factors, 1);
  int64_t output_height = osize[0];
  int64_t output_width = osize[1];
  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);
  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);
  std::vector<int64_t> outputSizes{
      nbatch, channels, output_height, output_width};
  MPSImage* X = imageFromTensor(input);
  MetalTensor mt{outputSizes};
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  mt.texture()->allocateTemporaryTextureStorage(outputSizes, commandBuffer);
  MPSImage* Y = imageFromMetalTensor(mt);
  if (@available(iOS 11.0, *)) {
    MPSCNNUpsamplingNearest* kernel = [[MPSCNNUpsamplingNearest alloc]
        initWithDevice:[MPSCNNContext sharedInstance].device
        integerScaleFactorX:(NSUInteger)scale_w.value()
        integerScaleFactorY:(NSUInteger)scale_h.value()];
    [kernel encodeToCommandBuffer:commandBuffer.buffer
                      sourceImage:X
                 destinationImage:Y];
  } else {
    NSUInteger sh = scale_h.value() * 10000;
    NSUInteger sw = scale_w.value() * 10000;
    id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
        specializedPipelineState:kernelFor(
                                     Y,
                                     @"resize_nearest",
                                     @"resize_nearest_nonarray")
                       Constants:@[
                         @(output_height),
                         @(output_width),
                         @(sh),
                         @(sw)
                       ]];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer.buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    const auto& launchParams = spatialPointwiseKernelLaunchParams(state, Y);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    [X markRead];
    [Y markRead];
  }
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

Tensor flatten_using_ints(
    const Tensor& input,
    int64_t start_dim,
    int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, input.dim());
  end_dim = maybe_wrap_dim(end_dim, input.dim());
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");
  std::vector<int64_t> shape;
  if (input.dim() == 0) {
    return input.reshape({1});
  }
  if (start_dim == end_dim) {
    return input;
  }
  const auto slice_numel =
      c10::multiply_integers(input.sizes().slice(start_dim, end_dim - start_dim + 1));
  shape.reserve(input.dim() - end_dim + start_dim);
  for (int64_t i = 0; i < start_dim; i++) {
    shape.push_back(input.size(i));
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < input.dim(); i++) {
    shape.push_back(input.size(i));
  }
  return input.reshape(shape);
}

Tensor cat_batch(const TensorList tensors, MetalTensor& mt) {
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(tensor);
  MPSImage* Y = imageFromMetalTensor(mt);

  ushort cat_dim4_pointer = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = commandBufferFromInputTensor(t);
    TORCH_CHECK([commandBuffer isEqual:Xcb], @"inputs have different command buffer");
    id<MTLComputeCommandEncoder> encoder = [commandBuffer.buffer computeCommandEncoder];
    id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
        pipelineState:metal::mpscnn::kernelFor(
                                     X, @"copy_offset", @"copy_offset_nonarray")];
    id<MTLBuffer> offsetBuffer = [[MPSCNNContext sharedInstance].device
        newBufferWithLength:1 * sizeof(ushort)
                    options:MTLResourceOptionCPUCacheModeWriteCombined];
    ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
    offsetBufferPtr[0] = cat_dim4_pointer;

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    [encoder setBuffer:offsetBuffer offset:0 atIndex:0];

    const auto& launchParams =
        metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    [X markRead];

    cat_dim4_pointer += t.size(0)*((t.size(1) + 3)/4);
  }

  auto output = MetalTensor::toTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat_feature(const TensorList tensors, MetalTensor& mt) {
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(tensor);
  MPSImage* Y = imageFromMetalTensor(mt);

  ushort channel_offset = 0;
  ushort channel4_offset = 0;
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    MPSImage* X = imageFromTensor(t);
    MetalCommandBuffer* Xcb = commandBufferFromInputTensor(t);
    TORCH_CHECK([commandBuffer isEqual:Xcb], @"inputs have different command buffer");
    id<MTLComputeCommandEncoder> encoder = [commandBuffer.buffer computeCommandEncoder];
    auto kernelString = metal::mpscnn::kernelFor(
                         X, @"append_features_off0", @"append_features_off0_nonarray");
    ushort tex_offset = channel_offset%4;
    if (tex_offset == 1) {
      kernelString = metal::mpscnn::kernelFor(
                         X, @"append_features_off1", @"append_features_off1_nonarray");
    }
    else if (tex_offset == 2) {
      kernelString = metal::mpscnn::kernelFor(
                         X, @"append_features_off2", @"append_features_off2_nonarray");
    }
    else if (tex_offset == 3) {
      kernelString = metal::mpscnn::kernelFor(
                         X, @"append_features_off3", @"append_features_off3_nonarray");
    }

    id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
        pipelineState:kernelString];
    id<MTLBuffer> offsetBuffer = [[MPSCNNContext sharedInstance].device
        newBufferWithLength:5 * sizeof(ushort)
                    options:MTLResourceOptionCPUCacheModeWriteCombined];
    ushort* offsetBufferPtr = (ushort*)[offsetBuffer contents];
    offsetBufferPtr[0] = (X.featureChannels + tex_offset + 3)/4;
    offsetBufferPtr[1] = (Y.featureChannels + 3)/4;
    offsetBufferPtr[2] = channel_offset/4;
    offsetBufferPtr[3] = (X.featureChannels + 3)/4;
    offsetBufferPtr[4] = X.numberOfImages*offsetBufferPtr[0];

    [encoder setComputePipelineState:state];
    [encoder setTexture:[X texture] atIndex:0];
    [encoder setTexture:[Y texture] atIndex:1];
    [encoder setBuffer:offsetBuffer offset:0 atIndex:0];

    ushort featureChannels = X.featureChannels;
    if (channel_offset%4 > 0) {
      featureChannels += tex_offset;
    }
    const auto& launchParams =
        metal::mpscnn::spatialPointwiseKernelLaunchParams(
          state,
          X.numberOfImages,
          featureChannels,
          X.height,
          X.width);
    [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
            threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
    [encoder endEncoding];
    [X markRead];

    channel4_offset += X.featureChannels/4;
    channel_offset += X.featureChannels;
  }

  auto output = MetalTensor::toTensor(std::move(mt), tensor.options());
  return output;
}

Tensor cat(const TensorList tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT(
      dim == 0 || dim == 1,
      "Metal cat is implemented only for batch dimension");
  int64_t cat_dim_size = 0;
  at::Tensor tensor = tensors[0];
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(tensor);
  for (int i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    TORCH_INTERNAL_ASSERT(
        t.dim() == 4, "Metal cat expects 4 dimensional inputs");
    TORCH_INTERNAL_ASSERT(t.is_metal(), "Metal cat expects metal tensors");

    for (int d = 0; d < 4; ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Metal cat inputs must have matching sizes except concatenated dimension");
    }
    cat_dim_size += t.size(dim);
  }
  auto result_size = tensor.sizes().vec();
  result_size[dim] = cat_dim_size;
  TORCH_INTERNAL_ASSERT(result_size[0] * ((result_size[1] + 3)/4) > 1, "Output tensor must be a texture array");
  MetalTensor mt{result_size};
  mt.texture()->setCommandBuffer(commandBuffer);
  mt.texture()->allocateTemporaryTextureStorage(result_size, commandBuffer);

  if (dim == 1) {
    return cat_feature(tensors, mt);
  }
  return cat_batch(tensors, mt);
}

Tensor copy_to_host(const Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  auto&& sizes = [X sizes];
  MetalTensor mt{sizes};
  mt.texture()->setCommandBuffer(commandBuffer);
  mt.texture()->allocateTextureStorage(sizes);
  MPSImage* Y = imageFromMetalTensor(mt);
  id<MTLComputeCommandEncoder> encoder =
      [commandBuffer.buffer computeCommandEncoder];
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
      specializedPipelineState:metal::mpscnn::kernelFor(
                                   X, @"copy", @"copy_nonarray")
                     Constants:@[
                       @(X.featureChannels),
                       @(X.height),
                       @(X.width)
                     ]];

  [encoder setComputePipelineState:state];
  [encoder setTexture:[X texture] atIndex:0];
  [encoder setTexture:[Y texture] atIndex:1];

  const auto& launchParams =
      metal::mpscnn::spatialPointwiseKernelLaunchParams(state, X);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  [X markRead];
  auto output = MetalTensor::toTensor(std::move(mt), input.options());
  return output;
}

}
}
}
}
