#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalTensorImpl.h>
#import <ATen/native/metal/MetalTensorImplStorage.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
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

Tensor binaryElementwiseShaderKernel(
    const Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* cb1 = getCommandBufferFromTensor(input1);
  MetalCommandBuffer* cb2 = getCommandBufferFromTensor(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = mt.texture()->image();
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
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
  [X1 markRead];
  [X2 markRead];
  auto output = makeTensor(std::move(mt), input1.options());
  return output;
}

Tensor& binaryElementwiseShaderKernel_(
    Tensor& input1,
    const Tensor& input2,
    NSString* arrayKernel,
    NSString* nonarrayKernel) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes().vec();
  }
  MetalCommandBuffer* cb1 = getCommandBufferFromTensor(input1);
  MetalCommandBuffer* cb2 = getCommandBufferFromTensor(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  MPSImage* Y = createTemporaryImage(cb1, outputSize);
  id<MTLComputePipelineState> state = [[MPSCNNContext sharedInstance]
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
  [X1 markRead];
  [X2 markRead];
  MetalTensorImpl* impl = (MetalTensorImpl*)input1.unsafeGetTensorImpl();
  MetalTensorImplStorage& implStorage = impl->unsafe_opaque_handle();
  implStorage.texture()->copyFromTexture(Y);
  return input1;
}

template <typename T>
Tensor binaryElementwiseMPSCNNKernel(
    const Tensor& input1,
    const Tensor& input2) {
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* cb1 = getCommandBufferFromTensor(input1);
  MetalCommandBuffer* cb2 = getCommandBufferFromTensor(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = mt.texture()->image();
  T* kernel = [[T alloc] initWithDevice:[MPSCNNContext sharedInstance].device];
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
  MPSImage* X1 = imageFromTensor(input1);
  MPSImage* X2 = imageFromTensor(input2);
  TORCH_CHECK(X1.numberOfImages == X2.numberOfImages &&
              X1.featureChannels == X2.featureChannels)
  std::vector<int64_t> outputSize = input1.sizes().vec();
  if (broadCastFirstInput(X1, X2)) {
    outputSize = input2.sizes().vec();
  }
  MetalTensorImplStorage mt{outputSize};
  MetalCommandBuffer* cb1 = getCommandBufferFromTensor(input1);
  MetalCommandBuffer* cb2 = getCommandBufferFromTensor(input2);
  TORCH_CHECK(
      [cb1 isEqual:cb2], @"inputs have different Metal command buffers");
  mt.texture()->allocateTemporaryTextureStorage(outputSize, cb1);
  MPSImage* Y = mt.texture()->image();
  T* kernel = [[T alloc] initWithDevice:[MPSCNNContext sharedInstance].device];
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
  implStorage.texture()->copyFromTexture(Y);
  return input1;
}

Tensor add_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNAdd>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, @"elementwise_add", @"elementwise_add_nonarray");
  }
}

Tensor& add__Tensor(Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNAdd>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, @"elementwise_add", @"elementwise_add_nonarray");
  }
}

Tensor sub_Tensor(const Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNSubtract>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, @"elementwise_sub", @"elementwise_sub_nonarray");
  }
}

Tensor& sub__Tensor(Tensor& input1, const Tensor& input2, const Scalar& alpha) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNSubtract>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, @"elementwise_sub", @"elementwise_sub_nonarray");
  }
}

Tensor mul_Tensor(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNMultiply>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, @"elementwise_mul", @"elementwise_mul_nonarray");
  }
}

Tensor& mul__Tensor(Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNMultiply>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, @"elementwise_mul", @"elementwise_mul_nonarray");
  }
}

Tensor div_Tensor(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel<MPSCNNDivide>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel(
        input1, input2_, @"elementwise_div", @"elementwise_div_nonarray");
  }
}

Tensor& div__Tensor(Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(input1.is_metal());
  TORCH_CHECK(input1.dim() == input2.dim());
  auto input2_ = input2.is_metal() ? input2 : input2.metal();
  if (@available(iOS 11.3, *)) {
    return binaryElementwiseMPSCNNKernel_<MPSCNNDivide>(input1, input2_);
  } else {
    return binaryElementwiseShaderKernel_(
        input1, input2_, @"elementwise_div", @"elementwise_div_nonarray");
  }
}

TORCH_LIBRARY_IMPL(aten, Metal, m) {
  m.impl("add.Tensor", TORCH_FN(add_Tensor));
  m.impl("add_.Tensor", TORCH_FN(add__Tensor));
  m.impl("mul.Tensor", TORCH_FN(mul_Tensor));
  m.impl("mul_.Tensor", TORCH_FN(mul__Tensor));
  m.impl("sub.Tensor", TORCH_FN(sub_Tensor));
  m.impl("sub_.Tensor", TORCH_FN(sub__Tensor));
  m.impl("div.Tensor", TORCH_FN(div_Tensor));
  m.impl("div_.Tensor", TORCH_FN(div__Tensor));
};

}
}
}
