#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalTensorUtils.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSCNNUtils.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>
#import <ATen/native/metal/mpscnn/MPSImageUtils.h>

#include <ATen/Tensor.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace metal {

using MetalTensorImpl = at::MetalTensorImpl<MetalTensorImplStorage>;

static inline bool broadCastFirstInput(MPSImage* X1, MPSImage* X2) {
  if ((X2.height > 1 && X1.height == 1) ||
      (X2.width > 1 && X1.width == 1)) {
    return true;
  }
  return false;
}

static inline void checkInputs(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      channelsSize(input1) == channelsSize(input2),
      "Metal binary elementwise ops require channel dimension to be equal!");
  if (batchSize(input1) != batchSize(input2)) {
    TORCH_CHECK(
        channelsSize(input1) % 4 == 0,
        "Metal binary elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  const uint32_t input1_h = heightSize(input1);
  const uint32_t input1_w = widthSize(input1);
  const uint32_t input2_h = heightSize(input2);
  const uint32_t input2_w = widthSize(input2);

  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Metal binary elementwise op!";
  if (input1_h != input2_h) {
    if (input1_h > input2_h) {
      TORCH_CHECK(input2_h == 1, broadcast_error_msg);
      TORCH_CHECK(input2_w == input1_w || input2_w == 1, broadcast_error_msg);
    } else if (input2_h > input1_h) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
      TORCH_CHECK(input1_w == input2_w || input1_w == 1, broadcast_error_msg);
    }
  } else if (input1_w != input2_w) {
    if (input1_w > input2_w) {
      TORCH_CHECK(input2_w == 1, broadcast_error_msg);
    } else if (input2_w > input1_w) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
    }
  }
}

static Tensor binaryElementwiseShaderKernel(
    const Tensor& input1,
    const Tensor& input2,
    const std::string& arrayKernel,
    const std::string& nonarrayKernel) {
  checkInputs(input1, input2);
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  IntArrayRef outputSize = input1.sizes();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes();
  }
  if(c10::multiply_integers(outputSize) == 0){
    return makeTensor({outputSize.vec()}, input1.options());
  }
  MetalTensorImplStorage mt{outputSize.vec()};
  MetalCommandBuffer* cb1 = getCommandBuffer(input1);
  MetalCommandBuffer* cb2 = getCommandBuffer(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  mt.texture()->allocateTemporaryStorage(outputSize, cb1);
  MPSImage* Y = mt.texture()->image();
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      pipelineState:mpscnn::kernelFor(X1, arrayKernel, nonarrayKernel)];
  id<MTLComputeCommandEncoder> encoder = [cb1.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X1 texture] atIndex:0];
  [encoder setTexture:[X2 texture] atIndex:1];
  [encoder setTexture:[Y texture] atIndex:2];
  const auto& launchParams =
      mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  auto output = makeTensor(std::move(mt), input1.options());
  return output;
}

static Tensor& binaryElementwiseShaderKernel_(
    Tensor& input1,
    const Tensor& input2,
    const std::string& arrayKernel,
    const std::string& nonarrayKernel) {
  checkInputs(input1, input2);
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  IntArrayRef outputSize = input1.sizes();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes();
  }
  if(c10::multiply_integers(outputSize) == 0){
      return input1;
  }
  MetalCommandBuffer* cb1 = getCommandBuffer(input1);
  MetalCommandBuffer* cb2 = getCommandBuffer(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  MPSImage* Y = createTemporaryImage(cb1, outputSize.vec());
  id<MTLComputePipelineState> state = [[MetalContext sharedInstance]
      pipelineState:mpscnn::kernelFor(X1, arrayKernel, nonarrayKernel)];
  id<MTLComputeCommandEncoder> encoder = [cb1.buffer computeCommandEncoder];
  [encoder setComputePipelineState:state];
  [encoder setTexture:[X1 texture] atIndex:0];
  [encoder setTexture:[X2 texture] atIndex:1];
  [encoder setTexture:[Y texture] atIndex:2];
  const auto& launchParams =
      mpscnn::spatialPointwiseKernelLaunchParams(state, Y);
  [encoder dispatchThreadgroups:launchParams.threadgroupsPerGrid
          threadsPerThreadgroup:launchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  MetalTensorImpl* impl = (MetalTensorImpl*)input1.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->setImage(Y);
  return input1;
}

template <typename T>
Tensor binaryElementwiseMPSCNNKernel(
    const Tensor& input1,
    const Tensor& input2) {
  checkInputs(input1, input2);
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  IntArrayRef outputSize = input1.sizes();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes();
  }
  if(c10::multiply_integers(outputSize) == 0){
      return makeTensor({outputSize.vec()}, input1.options());
  }
  MetalTensorImplStorage mt{outputSize.vec()};
  MetalCommandBuffer* cb1 = getCommandBuffer(input1);
  MetalCommandBuffer* cb2 = getCommandBuffer(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  mt.texture()->allocateTemporaryStorage(outputSize, cb1);
  MPSImage* Y = mt.texture()->image();
  T* kernel = [[T alloc] initWithDevice:[MetalContext sharedInstance].device];
  kernel.primaryStrideInPixelsY = X1.height == 1 ? 0 : 1;
  kernel.primaryStrideInPixelsX = X1.width == 1 ? 0 : 1;
  kernel.secondaryStrideInPixelsY = X2.height == 1 ? 0 : 1;
  kernel.secondaryStrideInPixelsX = X2.width == 1 ? 0 : 1;
  [kernel encodeToCommandBuffer:cb1.buffer
                   primaryImage:X1
                 secondaryImage:X2
               destinationImage:Y];
  auto output = makeTensor(std::move(mt), input1.options());
  return output;
}

template <typename T>
Tensor& binaryElementwiseMPSCNNKernel_(Tensor& input1, const Tensor& input2) {
  checkInputs(input1, input2);
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  IntArrayRef outputSize = input1.sizes();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes();
  }
  if(c10::multiply_integers(outputSize) == 0){
    return input1;
  }
  MetalCommandBuffer* cb1 = getCommandBuffer(input1);
  MetalCommandBuffer* cb2 = getCommandBuffer(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  MPSImage* Y = createTemporaryImage(cb1, outputSize.vec());
  T* kernel = [[T alloc] initWithDevice:[MetalContext sharedInstance].device];
  kernel.primaryStrideInPixelsY = X1.height == 1 ? 0 : 1;
  kernel.primaryStrideInPixelsX = X1.width == 1 ? 0 : 1;
  kernel.secondaryStrideInPixelsY = X2.height == 1 ? 0 : 1;
  kernel.secondaryStrideInPixelsX = X2.width == 1 ? 0 : 1;
  [kernel encodeToCommandBuffer:cb1.buffer
                   primaryImage:X1
                 secondaryImage:X2
               destinationImage:Y];
  MetalTensorImpl* impl = (MetalTensorImpl*)input1.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->setImage(Y);
  return input1;
}

static Tensor add_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNAdd>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, "elementwise_add", "elementwise_add_nonarray");
  }
}

static Tensor& add__Tensor(Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNAdd>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, "elementwise_add", "elementwise_add_nonarray");
  }
}

static Tensor sub_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNSubtract>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, "elementwise_sub", "elementwise_sub_nonarray");
  }
}

static Tensor& sub__Tensor(Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNSubtract>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, "elementwise_sub", "elementwise_sub_nonarray");
  }
}

static Tensor mul_Tensor(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNMultiply>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, "elementwise_mul", "elementwise_mul_nonarray");
  }
}

static Tensor& mul__Tensor(Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNMultiply>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, "elementwise_mul", "elementwise_mul_nonarray");
  }
}

static Tensor div_Tensor(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNDivide>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, "elementwise_div", "elementwise_div_nonarray");
  }
}

static Tensor& div__Tensor(Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNDivide>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, "elementwise_div", "elementwise_div_nonarray");
  }
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add__Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul__Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub__Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_Tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div__Tensor));
};

}
}
}
