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
  [pool setOffset:{.x = static_cast<NSInteger>(kernel_size[0] / 2),
                   .y = static_cast<NSInteger>(kernel_size[1] / 2),
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

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor binaryElementwiseKernel(
    const Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
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
Tensor& binaryElementwiseKernel_(
    Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  std::vector<int64_t> outputSize = input1.sizes().vec();
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

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor add(const Tensor& input1, const Tensor& input2) {
  return binaryElementwiseKernel(
      input1, input2, @"elementwise_add", @"elementwise_add_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor& add_(Tensor& input1, const Tensor& input2) {
  return binaryElementwiseKernel_(
      input1, input2, @"elementwise_add", @"elementwise_add_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor sub(const Tensor& input1, const Tensor& input2) {
  return binaryElementwiseKernel(
      input1, input2, @"elementwise_sub", @"elementwise_sub_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor mul(const Tensor& input1, const Tensor& input2) {
  return binaryElementwiseKernel(
      input1, input2, @"elementwise_mul", @"elementwise_mul_nonarray");
}

API_AVAILABLE(ios(10.0), macos(10.13))
Tensor t(const Tensor& input) {
  auto strides = input.strides().vec();
  auto sizes = input.sizes().vec();
  MPSImage* X = imageFromTensor(input);
  TORCH_CHECK(X.numberOfImages == 1);
  TORCH_CHECK(X.featureChannels == 1);
  MetalTensor mt({sizes[1], sizes[0]}, {strides[1], strides[0]});
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
  auto slice_numel =
      prod_intlist(input.sizes().slice(start_dim, end_dim - start_dim + 1));
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

Tensor copy_to_host(const Tensor& input) {
  MPSImage* X = imageFromTensor(input);
  MetalCommandBuffer* commandBuffer = commandBufferFromInputTensor(input);
  auto&& sizes = [X sizes];
  auto dummy = at::zeros(input.sizes()).contiguous();
  auto strides = dummy.strides();
  MetalTensor mt{sizes, strides.vec()};
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
